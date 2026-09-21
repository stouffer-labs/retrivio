//! Project discovery and the corpus walk: index scope and scoped refresh planning, root discovery, scan caps, file selection, the collector, the file manifest and text chunking.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{params, Connection};
use sha1::{Digest, Sha1};
use xxhash_rust::xxh64;

use crate::config::{selection_tier, ConfigValues, ScanSettings};
use crate::db::{list_tracked_roots_full_conn, TrackedRoot};
#[cfg(test)]
use crate::index::INJECT_COLLECTOR_PANIC;
use crate::index::{get_project_by_path, ProjectChunk, ProjectCorpus, ProjectDoc, ScannedFile};
use crate::util::{
    collapse_whitespace, file_mtime, metadata_mtime, metadata_mtime_ns, normalize_path, now_ts,
    path_is_under_any, word_tokens,
};
use crate::watch::{longest_prefix_match, path_depth};
use crate::{code_intel, documents};

/// What one indexing run covers.
///
/// Discovery (turning a root directory into project directories) only ever runs on roots.
/// Project directories are taken as given: a project with several child directories and no
/// marker file looks like a workspace to discovery and would otherwise be split into one
/// duplicate "project" row per child.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IndexScope {
    /// Every tracked root with full discovery (sweeps, `index`, bare `refresh`).
    AllRoots,
    /// Discovery on exactly these root directories (tracked or ad hoc) plus exactly these
    /// project directories, re-collected as they are.
    Targets {
        roots: Vec<PathBuf>,
        projects: Vec<PathBuf>,
    },
}

impl IndexScope {
    pub(crate) fn roots(roots: Vec<PathBuf>) -> Self {
        Self::Targets {
            roots,
            projects: Vec::new(),
        }
    }

    pub(crate) fn projects(projects: Vec<PathBuf>) -> Self {
        Self::Targets {
            roots: Vec::new(),
            projects,
        }
    }

    /// True when a scoped run has nothing to do.
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::AllRoots => false,
            Self::Targets { roots, projects } => roots.is_empty() && projects.is_empty(),
        }
    }

    /// Every directory named by the scope, roots first.
    pub(crate) fn target_paths(&self) -> Vec<PathBuf> {
        match self {
            Self::AllRoots => Vec::new(),
            Self::Targets { roots, projects } => {
                roots.iter().chain(projects.iter()).cloned().collect()
            }
        }
    }
}

/// Resolve an [`IndexScope`] into the tracked roots (needed for per-project excludes) and the
/// exact project directories the run covers.
/// The roots and projects of a run, plus the roots whose discovery listing was incomplete
/// (their projects are protected from removal).
pub(crate) fn resolve_index_targets(
    conn: &Connection,
    cfg: &ConfigValues,
    scope: &IndexScope,
) -> Result<IndexTargets, String> {
    let mut roots = resolve_roots(conn, cfg, None)?;
    let settings = ScanSettings::from_cfg(cfg);
    match scope {
        IndexScope::AllRoots => {
            let discovery = discover_projects_full(&roots, &settings);
            Ok(IndexTargets {
                roots,
                projects: discovery.projects,
                incomplete_roots: discovery.incomplete_roots,
                shallow: discovery.shallow,
            })
        }
        IndexScope::Targets {
            roots: root_paths,
            projects: project_paths,
        } => {
            let scoped = resolve_roots(conn, cfg, Some(root_paths.clone()))?;
            let discovery = discover_projects_full(&scoped, &settings);
            let mut projects = discovery.projects;
            let mut shallow = discovery.shallow;
            for root in scoped {
                if !roots.iter().any(|t| t.path == root.path) {
                    roots.push(root);
                }
            }
            for raw in project_paths {
                let project = normalize_path(&raw.to_string_lossy());
                if project.is_dir() && !projects.contains(&project) {
                    // A tracked root named as a project is its root-files project when
                    // discovery splits that root; otherwise the root is one recursive project.
                    if let Some(tracked) = roots.iter().find(|t| t.path == project) {
                        if discover_root(&tracked.path, &tracked.absolute_excludes(), &settings)
                            .root_files
                            .is_some()
                        {
                            shallow.insert(project.clone());
                        }
                    }
                    projects.push(project);
                }
            }
            Ok(IndexTargets {
                roots,
                projects,
                incomplete_roots: HashSet::new(),
                shallow,
            })
        }
    }
}

/// What one index run covers.
#[derive(Default)]
pub(crate) struct IndexTargets {
    /// Tracked roots (for per-project excludes and root protection).
    pub(crate) roots: Vec<TrackedRoot>,
    /// The exact project directories, root-files projects included.
    pub(crate) projects: Vec<PathBuf>,
    /// Roots whose discovery listing was incomplete; their projects are protected from removal.
    pub(crate) incomplete_roots: HashSet<PathBuf>,
    /// Projects scanned shallow (a root's top-level files only).
    pub(crate) shallow: HashSet<PathBuf>,
}

/// Interpret the paths given to a scoped refresh (`retrivio refresh <path>`, `POST /refresh`,
/// MCP `run_forced_refresh`).
///
/// A tracked root gets discovery, as a full run would. A project (discovery of its tracked
/// root yields it, or a `projects` row exists for it) is re-collected as that one project and
/// never discovered into sub-projects. Any other path is an error naming the project or root
/// to refresh instead.
pub(crate) fn plan_scoped_refresh(
    conn: &Connection,
    cfg: &ConfigValues,
    paths: &[PathBuf],
) -> Result<IndexScope, String> {
    let tracked = resolve_roots(conn, cfg, None)?;
    let settings = ScanSettings::from_cfg(cfg);
    let root_paths: Vec<PathBuf> = tracked.iter().map(|r| r.path.clone()).collect();
    let mut discovered_by_root: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut projects: Vec<PathBuf> = Vec::new();

    for raw in paths {
        let path = normalize_path(&raw.to_string_lossy());
        if !path.is_dir() {
            return Err(format!("{} is not a directory", path.display()));
        }
        if root_paths.contains(&path) {
            if !roots.contains(&path) {
                roots.push(path);
            }
            continue;
        }
        let root = longest_prefix_match(&path, &root_paths).cloned();
        let discovered: Vec<PathBuf> = match &root {
            Some(r) => discovered_by_root
                .entry(r.clone())
                .or_insert_with(|| {
                    tracked
                        .iter()
                        .find(|t| t.path == *r)
                        .map(|t| discover_root_projects(&t.path, &t.absolute_excludes(), &settings))
                        .unwrap_or_default()
                })
                .clone(),
            None => Vec::new(),
        };
        if discovered.contains(&path) {
            if !projects.contains(&path) {
                projects.push(path);
            }
            continue;
        }
        let containing = discovered
            .iter()
            .filter(|q| path.starts_with(q))
            .max_by_key(|q| path_depth(q))
            .cloned();
        if get_project_by_path(conn, &path.to_string_lossy())?.is_some() {
            if let Some(q) = &containing {
                eprintln!(
                    "note: {} is indexed as its own project but discovery now places it inside {}; `retrivio prune` will remove the extra row",
                    path.display(),
                    q.display()
                );
            }
            if !projects.contains(&path) {
                projects.push(path);
            }
            continue;
        }
        let mut msg = format!(
            "{} is neither a tracked root nor a discovered project",
            path.display()
        );
        match (&root, &containing) {
            (_, Some(q)) => msg.push_str(&format!(
                "; it is part of project {0}. Run `retrivio refresh {0}` instead",
                q.display()
            )),
            (Some(r), None) => msg.push_str(&format!(
                " under tracked root {0}. Run `retrivio refresh {0}` to refresh that root",
                r.display()
            )),
            (None, None) => msg.push_str(&format!(
                ". It is not under any tracked root; run `retrivio add {}` to track it",
                path.display()
            )),
        }
        return Err(msg);
    }
    Ok(IndexScope::Targets { roots, projects })
}

#[cfg(test)]
mod scoped_refresh_tests {
    use super::*;
    use crate::config::{ConfigValues, ScanSettings};
    use crate::db::{ensure_tracked_root_conn, init_schema};
    use crate::index::begin_project_update;
    use crate::util::normalize_path;
    use crate::watch::derive_watch_targets;
    use rusqlite::Connection;
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::{Path, PathBuf};

    /// A workspace root under the repo `tmp/` with two projects, each holding two child
    /// directories and no marker file, so discovery on a project would split it.
    fn workspace(name: &str) -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("refresh-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for (project, subs) in [
            ("proj-a", ["notes", "reports"]),
            ("proj-b", ["src", "plans"]),
        ] {
            for sub in subs {
                let dir = root.join(project).join(sub);
                fs::create_dir_all(&dir).expect("create sub dir");
                fs::write(dir.join("readme.md"), format!("{} {}", project, sub)).expect("write");
            }
            fs::write(root.join(project).join("README.md"), project).expect("write readme");
        }
        normalize_path(&root.to_string_lossy())
    }

    fn conn_tracking(root: &Path) -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        init_schema(&conn).expect("init schema");
        ensure_tracked_root_conn(&conn, root, 0.0).expect("track root");
        conn
    }

    fn cfg() -> ConfigValues {
        ConfigValues::from_map(HashMap::new())
    }

    #[test]
    fn refreshing_a_project_path_targets_exactly_that_project() {
        let root = workspace("project");
        let conn = conn_tracking(&root);
        let a = root.join("proj-a");

        let scope = plan_scoped_refresh(&conn, &cfg(), std::slice::from_ref(&a)).expect("plan");
        assert_eq!(scope, IndexScope::projects(vec![a.clone()]));
        let IndexTargets {
            roots, projects, ..
        } = resolve_index_targets(&conn, &cfg(), &scope).expect("resolve");
        assert_eq!(
            projects,
            vec![a.clone()],
            "only the project itself, never its children"
        );
        assert_eq!(roots.len(), 1);

        // prune's stale-row criterion: a row survives when discovery of the tracked roots
        // yields its path. Every target here is such a path, so prune finds nothing to remove.
        let discovered = discover_projects_full(&roots, &ScanSettings::default()).projects;
        assert!(projects.iter().all(|p| discovered.contains(p)));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn refreshing_the_root_targets_both_projects() {
        let root = workspace("root");
        let conn = conn_tracking(&root);

        let scope = plan_scoped_refresh(&conn, &cfg(), std::slice::from_ref(&root)).expect("plan");
        assert_eq!(scope, IndexScope::roots(vec![root.clone()]));
        let IndexTargets { projects, .. } =
            resolve_index_targets(&conn, &cfg(), &scope).expect("resolve");
        assert_eq!(projects, vec![root.join("proj-a"), root.join("proj-b")]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn refreshing_a_child_directory_or_untracked_path_is_an_error_naming_the_alternative() {
        let root = workspace("errors");
        let conn = conn_tracking(&root);
        let a = root.join("proj-a");

        let err = plan_scoped_refresh(&conn, &cfg(), &[a.join("reports")]).unwrap_err();
        assert!(
            err.contains("neither a tracked root nor a discovered project"),
            "{}",
            err
        );
        assert!(
            err.contains(&format!("retrivio refresh {}", a.display())),
            "{}",
            err
        );

        let outside = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("refresh-outside-{}", std::process::id()));
        fs::create_dir_all(&outside).expect("create outside dir");
        let err = plan_scoped_refresh(&conn, &cfg(), std::slice::from_ref(&outside)).unwrap_err();
        assert!(err.contains("not under any tracked root"), "{}", err);
        assert!(err.contains("retrivio add"), "{}", err);

        let err = plan_scoped_refresh(&conn, &cfg(), &[root.join("missing")]).unwrap_err();
        assert!(err.contains("is not a directory"), "{}", err);
        let _ = fs::remove_dir_all(&outside);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_indexed_project_row_is_refreshed_even_when_discovery_does_not_list_it() {
        let root = workspace("indexed");
        let conn = conn_tracking(&root);
        let stale = root.join("proj-a").join("reports");
        begin_project_update(&conn, &stale.to_string_lossy(), "reports").expect("row");

        let scope = plan_scoped_refresh(&conn, &cfg(), std::slice::from_ref(&stale)).expect("plan");
        assert_eq!(scope, IndexScope::projects(vec![stale.clone()]));
        let IndexTargets { projects, .. } =
            resolve_index_targets(&conn, &cfg(), &scope).expect("resolve");
        assert_eq!(projects, vec![stale]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn mixed_paths_keep_roots_and_projects_apart_and_deduplicate() {
        let root = workspace("mixed");
        let conn = conn_tracking(&root);
        let a = root.join("proj-a");
        let scope = plan_scoped_refresh(&conn, &cfg(), &[a.clone(), root.clone(), a.clone()])
            .expect("plan");
        assert_eq!(
            scope,
            IndexScope::Targets {
                roots: vec![root.clone()],
                projects: vec![a.clone()]
            }
        );
        let IndexTargets { projects, .. } =
            resolve_index_targets(&conn, &cfg(), &scope).expect("resolve");
        assert_eq!(projects, vec![a, root.join("proj-b")]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn watch_events_inside_a_project_target_that_project_not_its_children() {
        let root = workspace("watch");
        let conn = conn_tracking(&root);
        let tracked = resolve_roots(&conn, &cfg(), None).expect("roots");
        let a = root.join("proj-a");
        let mut pending: HashSet<PathBuf> = HashSet::new();
        pending.insert(a.join("reports").join("readme.md"));
        pending.insert(a.join("notes").join("readme.md"));

        let scope = derive_watch_targets(&pending, &tracked, &ScanSettings::default());
        assert_eq!(scope, IndexScope::projects(vec![a.clone()]));
        let IndexTargets { projects, .. } =
            resolve_index_targets(&conn, &cfg(), &scope).expect("resolve");
        assert_eq!(projects, vec![a]);

        // A new directory is discovered at event time and targeted as a project itself.
        let fresh = root.join("proj-c");
        fs::create_dir_all(&fresh).expect("create proj-c");
        let mut pending: HashSet<PathBuf> = HashSet::new();
        pending.insert(fresh.join("readme.md"));
        let scope = derive_watch_targets(&pending, &tracked, &ScanSettings::default());
        assert_eq!(scope, IndexScope::projects(vec![fresh.clone()]));

        // A file under the root but inside no project falls back to discovery on the root.
        let mut pending: HashSet<PathBuf> = HashSet::new();
        pending.insert(root.join("notes.md"));
        let scope = derive_watch_targets(&pending, &tracked, &ScanSettings::default());
        assert_eq!(scope, IndexScope::roots(vec![root.clone()]));
        let _ = fs::remove_dir_all(&root);
    }
}

pub(crate) fn resolve_roots(
    conn: &Connection,
    _cfg: &ConfigValues,
    scope_roots: Option<Vec<PathBuf>>,
) -> Result<Vec<TrackedRoot>, String> {
    if let Some(roots) = scope_roots {
        // Ad-hoc roots from CLI: look up excludes from DB if the root is tracked,
        // otherwise use empty excludes.
        let all_tracked = list_tracked_roots_full_conn(conn)?;
        let mut out = Vec::new();
        for root in roots {
            let p = normalize_path(&root.to_string_lossy());
            if p.is_dir() {
                let excludes = all_tracked
                    .iter()
                    .find(|t| t.path == p)
                    .map(|t| t.exclude_patterns.clone())
                    .unwrap_or_default();
                out.push(TrackedRoot {
                    path: p,
                    exclude_patterns: excludes,
                });
            }
        }
        return Ok(out);
    }

    let rows = list_tracked_roots_full_conn(conn)?;
    Ok(rows
        .into_iter()
        .map(|r| TrackedRoot {
            path: normalize_path(&r.path.to_string_lossy()),
            exclude_patterns: r.exclude_patterns,
        })
        .collect())
}

pub(crate) fn project_excludes_for_path(
    project_dir: &Path,
    roots: &[TrackedRoot],
) -> HashSet<PathBuf> {
    let project = normalize_path(&project_dir.to_string_lossy());
    let mut out: HashSet<PathBuf> = HashSet::new();
    for root in roots {
        if !(project == root.path || project.starts_with(&root.path)) {
            continue;
        }
        for abs in root.absolute_excludes() {
            if abs == project || abs.starts_with(&project) {
                out.insert(abs);
            }
        }
    }
    out
}

/// What discovery found under every tracked root.
#[derive(Clone, Debug, Default)]
pub(crate) struct Discovery {
    /// Every project path, root-files projects included, sorted by basename.
    pub(crate) projects: Vec<PathBuf>,
    /// The paths in `projects` that are root-files projects: a tracked root indexed for its
    /// own top-level files only (see [`RootDiscovery::root_files`]). Scanned shallow.
    shallow: HashSet<PathBuf>,
    /// Roots whose listing was incomplete (a directory entry could not be read). Projects
    /// under them may be missing from `projects`, so `index` and `prune` never remove project
    /// rows under them.
    pub(crate) incomplete_roots: HashSet<PathBuf>,
}

impl Discovery {
    pub(crate) fn is_shallow(&self, project: &Path) -> bool {
        self.shallow.contains(project)
    }
}

pub(crate) fn discover_projects_full(roots: &[TrackedRoot], settings: &ScanSettings) -> Discovery {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Discovery::default();

    for root in roots {
        if !root.path.is_dir() {
            continue;
        }
        let found = discover_root(&root.path, &root.absolute_excludes(), settings);
        if found.incomplete {
            out.incomplete_roots.insert(root.path.clone());
        }
        for candidate in found.projects {
            let key = candidate.to_string_lossy().to_string();
            if seen.insert(key) {
                out.projects.push(candidate);
            }
        }
        if let Some(root_files) = found.root_files {
            let key = root_files.to_string_lossy().to_string();
            if seen.insert(key) {
                out.shallow.insert(root_files.clone());
                out.projects.push(root_files);
            }
        }
    }

    out.projects
        .sort_by_key(|p| p.file_name().map(|s| s.to_string_lossy().to_lowercase()));
    out
}

/// What discovery found under one tracked root.
#[derive(Clone, Debug, Default)]
pub(crate) struct RootDiscovery {
    /// Project directories, each indexed recursively.
    projects: Vec<PathBuf>,
    /// The root itself as a project for the files lying directly under it, when the root is
    /// split into child projects and holds at least one indexable file of its own. Its path is
    /// the root's path, its title `<root basename> (root files)`, and it is scanned shallow:
    /// direct files only, no subdirectories. It takes part in index, prune and refresh like
    /// any project and disappears when its last file does.
    root_files: Option<PathBuf>,
    /// A directory listing on the way could not be read in full.
    incomplete: bool,
}

pub(crate) fn discover_root(
    root: &Path,
    exclude_abs: &HashSet<PathBuf>,
    settings: &ScanSettings,
) -> RootDiscovery {
    let (projects, incomplete) = discover_root_projects_checked(root, exclude_abs, settings);
    let root_path = normalize_path(&root.to_string_lossy());
    let root_files = (!projects.contains(&root_path)
        && has_direct_indexable_files(&root_path, settings))
    .then_some(root_path);
    RootDiscovery {
        projects,
        root_files,
        incomplete,
    }
}

/// True when `dir` holds, directly, at least one file the walk would index (same rules as
/// [`walk_project_files`]: a regular, non-hidden file with an indexable suffix and content).
pub(crate) fn has_direct_indexable_files(dir: &Path, settings: &ScanSettings) -> bool {
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    rd.flatten().any(|entry| {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            return false;
        }
        let Ok(ft) = entry.file_type() else {
            return false;
        };
        if !ft.is_file() {
            return false;
        }
        let ext = Path::new(&name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        settings.is_indexable_suffix(&format!(".{}", ext))
            && entry.metadata().map(|m| m.len() > 0).unwrap_or(false)
    })
}

/// Child directories of `root` (project candidates). `incomplete` is set when the listing
/// could not be read in full, so the caller never treats an unlisted project as gone.
pub(crate) fn list_project_child_dirs(
    root: &Path,
    exclude_abs: &HashSet<PathBuf>,
    incomplete: &mut bool,
    settings: &ScanSettings,
) -> Vec<PathBuf> {
    let mut children: Vec<PathBuf> = Vec::new();
    match fs::read_dir(root) {
        Ok(rd) => {
            for entry in rd {
                let Ok(entry) = entry else {
                    *incomplete = true;
                    continue;
                };
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.') || settings.is_skip_dir(&name) {
                    continue;
                }
                match entry.file_type() {
                    Ok(ft) if ft.is_dir() => {
                        let p = normalize_path(&entry.path().to_string_lossy());
                        if exclude_abs.contains(&p) {
                            continue;
                        }
                        children.push(p);
                    }
                    Ok(_) => {}
                    Err(_) => *incomplete = true,
                }
            }
        }
        Err(_) => *incomplete = true,
    }
    children.sort_by_key(|p| p.file_name().map(|s| s.to_string_lossy().to_lowercase()));
    children
}

pub(crate) fn discover_root_projects(
    root: &Path,
    exclude_abs: &HashSet<PathBuf>,
    settings: &ScanSettings,
) -> Vec<PathBuf> {
    discover_root_projects_checked(root, exclude_abs, settings).0
}

/// [`discover_root_projects`] plus whether any directory listing on the way was incomplete.
pub(crate) fn discover_root_projects_checked(
    root: &Path,
    exclude_abs: &HashSet<PathBuf>,
    settings: &ScanSettings,
) -> (Vec<PathBuf>, bool) {
    let mut incomplete = false;
    let projects = discover_root_projects_inner(root, exclude_abs, &mut incomplete, settings);
    (projects, incomplete)
}

pub(crate) fn discover_root_projects_inner(
    root: &Path,
    exclude_abs: &HashSet<PathBuf>,
    incomplete: &mut bool,
    settings: &ScanSettings,
) -> Vec<PathBuf> {
    // Unwrap common single-container roots (e.g. demo-data/projects/*) so users
    // can track the parent and still get project-level indexing.
    let mut cursor = normalize_path(&root.to_string_lossy());
    let container_names: HashSet<&str> = ["projects", "repos", "repositories", "workspaces"]
        .into_iter()
        .collect();
    for _ in 0..4 {
        let children = list_project_child_dirs(&cursor, exclude_abs, incomplete, settings);
        if children.is_empty() {
            return vec![cursor];
        }
        let mut expanded_from_containers: Vec<PathBuf> = Vec::new();
        for child in &children {
            let child_name = child
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if container_names.contains(child_name.as_str()) {
                let mut grand = list_project_child_dirs(child, exclude_abs, incomplete, settings);
                if grand.is_empty() {
                    expanded_from_containers.push(child.clone());
                } else {
                    expanded_from_containers.append(&mut grand);
                }
            }
        }
        if !expanded_from_containers.is_empty() {
            expanded_from_containers.sort_by_key(|p| {
                p.file_name()
                    .map(|s| s.to_string_lossy().to_lowercase())
                    .unwrap_or_default()
            });
            expanded_from_containers.dedup_by(|a, b| a == b);
            return expanded_from_containers;
        }
        if looks_like_single_project_root(&cursor) {
            return vec![cursor];
        }
        if looks_like_workspace_root(&cursor, &children) {
            return children;
        }
        if children.len() == 1 {
            let child_name = children[0]
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if container_names.contains(child_name.as_str()) {
                cursor = children[0].clone();
                continue;
            }
            if has_indexable_files_in_root(&cursor, settings)
                && is_common_single_project_child_dir(child_name.as_str())
            {
                return vec![cursor];
            }
        }
        if children.len() == 1 && !looks_like_single_project_root(&cursor) {
            cursor = children[0].clone();
            continue;
        }
        return vec![cursor];
    }
    vec![cursor]
}

pub(crate) fn looks_like_workspace_root(root: &Path, children: &[PathBuf]) -> bool {
    if children.len() < 2 {
        return false;
    }
    !looks_like_single_project_root(root)
}

pub(crate) fn looks_like_single_project_root(root: &Path) -> bool {
    let marker_files = [
        "cargo.toml",
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "makefile",
        "justfile",
    ];
    let Ok(rd) = fs::read_dir(root) else {
        return false;
    };
    let mut names: HashSet<String> = HashSet::new();
    for entry in rd.flatten() {
        if let Ok(ft) = entry.file_type() {
            if ft.is_file() {
                names.insert(entry.file_name().to_string_lossy().to_lowercase());
            }
        }
    }
    marker_files.iter().any(|marker| names.contains(*marker))
}

pub(crate) fn is_common_single_project_child_dir(name: &str) -> bool {
    matches!(
        name,
        "src" | "app" | "lib" | "cmd" | "tests" | "test" | "docs" | "scripts"
    )
}

#[cfg(test)]
mod project_discovery_tests {
    use super::*;
    use crate::cli::{
        current_platform_exe_name, preferred_local_repo_binary, repo_root_from_anchor_path,
    };
    use crate::config::{ConfigValues, ScanSettings};
    use crate::util::normalize_path;
    use std::collections::HashSet;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    use std::{fs, thread};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn extra_skip_dirs_from_config_are_honoured() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "skip_dir_names".to_string(),
            " zz-skip-me , zz-tmp,, ".to_string(),
        );
        let cfg = ConfigValues::from_map(map);
        let set = cfg.skip_dir_name_set();
        assert_eq!(set.len(), 2);
        assert!(set.contains("zz-skip-me") && set.contains("zz-tmp"));
        // The settings are a value built from the config; nothing process-wide.
        let settings = ScanSettings::from_cfg(&cfg);
        assert_eq!(settings.extra_skip_dirs, set);
        assert!(settings.is_skip_dir("zz-skip-me"));
        assert!(settings.is_skip_dir("zz-tmp"));
        assert!(
            settings.is_skip_dir("node_modules"),
            "built-ins still apply"
        );
        assert!(!settings.is_skip_dir("src"));
        assert!(!ScanSettings::default().is_skip_dir("zz-skip-me"));
        let other =
            ScanSettings::from_cfg(&ConfigValues::from_map(std::collections::HashMap::new()));
        assert!(
            !other.is_skip_dir("zz-skip-me"),
            "another run's settings are unaffected"
        );
        assert!(settings.is_skip_dir("zz-skip-me"));

        // Discovery consults the same predicate.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("test-skipdirs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("zz-skip-me")).expect("create skipped dir");
        fs::create_dir_all(root.join("keep-me")).expect("create kept dir");
        let children = list_project_child_dirs(&root, &HashSet::new(), &mut false, &settings);
        let names: Vec<String> = children
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .collect();
        assert_eq!(names, vec!["keep-me".to_string()]);
        let _ = fs::remove_dir_all(&root);
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        p.push(format!("retrivio-{}-{}-{}", prefix, std::process::id(), n));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).expect("create temp dir");
        p
    }

    fn write_executable(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, "#!/bin/sh\nexit 0\n").expect("write executable");
        #[cfg(unix)]
        {
            let mut perms = fs::metadata(path).expect("stat executable").permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).expect("chmod executable");
        }
    }

    #[test]
    fn discovers_children_inside_projects_container_even_with_root_readme() {
        let root = temp_dir("container-root");
        fs::write(root.join("README.md"), "root").expect("write readme");
        let projects = root.join("projects");
        fs::create_dir_all(projects.join("proj-a")).expect("mk proj-a");
        fs::create_dir_all(projects.join("proj-b")).expect("mk proj-b");
        fs::write(projects.join("proj-a").join("notes.md"), "a").expect("write proj-a");
        fs::write(projects.join("proj-b").join("notes.md"), "b").expect("write proj-b");

        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let got = discover_root_projects(&root, &no_excludes, &ScanSettings::default());
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|p| p.ends_with("proj-a")));
        assert!(got.iter().any(|p| p.ends_with("proj-b")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn keeps_single_repo_root_when_project_markers_exist() {
        let root = temp_dir("single-root");
        fs::write(root.join("Cargo.toml"), "[package]\nname='x'\n").expect("write cargo");
        fs::create_dir_all(root.join("src")).expect("mk src");
        fs::create_dir_all(root.join("docs")).expect("mk docs");
        fs::write(root.join("src").join("main.rs"), "fn main() {}").expect("write main");

        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let got = discover_root_projects(&root, &no_excludes, &ScanSettings::default());
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], normalize_path(&root.to_string_lossy()));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn expands_workspace_children_even_with_root_files() {
        let root = temp_dir("workspace-root");
        fs::write(root.join("README.md"), "workspace").expect("write readme");
        fs::create_dir_all(root.join("alpha")).expect("mk alpha");
        fs::create_dir_all(root.join("beta")).expect("mk beta");
        fs::write(root.join("alpha").join("notes.md"), "a").expect("write alpha");
        fs::write(root.join("beta").join("notes.md"), "b").expect("write beta");

        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let got = discover_root_projects(&root, &no_excludes, &ScanSettings::default());
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|p| p.ends_with("alpha")));
        assert!(got.iter().any(|p| p.ends_with("beta")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn excludes_skip_directories_during_discovery() {
        let root = temp_dir("exclude-test");
        fs::create_dir_all(root.join("alpha")).expect("mk alpha");
        fs::create_dir_all(root.join("beta")).expect("mk beta");
        fs::create_dir_all(root.join("gamma")).expect("mk gamma");
        fs::write(root.join("alpha").join("notes.md"), "a").expect("write alpha");
        fs::write(root.join("beta").join("notes.md"), "b").expect("write beta");
        fs::write(root.join("gamma").join("notes.md"), "c").expect("write gamma");

        // Without excludes: all 3 children
        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let got = discover_root_projects(&root, &no_excludes, &ScanSettings::default());
        assert_eq!(got.len(), 3);

        // With beta excluded
        let mut excludes: HashSet<PathBuf> = HashSet::new();
        excludes.insert(normalize_path(&root.join("beta").to_string_lossy()));
        let got = discover_root_projects(&root, &excludes, &ScanSettings::default());
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|p| p.ends_with("alpha")));
        assert!(got.iter().any(|p| p.ends_with("gamma")));
        assert!(!got.iter().any(|p| p.ends_with("beta")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn collect_project_corpus_respects_excluded_subdirs() {
        let root = temp_dir("corpus-exclude");
        fs::create_dir_all(root.join("src")).expect("mk src");
        fs::create_dir_all(root.join("tmp")).expect("mk tmp");
        fs::write(
            root.join("src").join("main.rs"),
            "fn keep_me() { println!(\"ok\"); }\n",
        )
        .expect("write src");
        fs::write(
            root.join("tmp").join("generated.rs"),
            "fn generated_artifact() { println!(\"x\"); }\n".repeat(120),
        )
        .expect("write tmp");

        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let caps = ScanCaps::default();
        let scan = project_scan(&root, &no_excludes, &caps, false, &ScanSettings::default());
        let all = collect_project_corpus(
            &root,
            &scan,
            &caps,
            100_000,
            &FileManifest::new(),
            true,
            &ScanSettings::default(),
        );
        assert!(all
            .chunks
            .iter()
            .any(|c| c.doc_rel_path.starts_with("tmp/")));

        let mut excludes: HashSet<PathBuf> = HashSet::new();
        excludes.insert(normalize_path(&root.join("tmp").to_string_lossy()));
        let scan_excluded = project_scan(&root, &excludes, &caps, false, &ScanSettings::default());
        let filtered = collect_project_corpus(
            &root,
            &scan_excluded,
            &caps,
            100_000,
            &FileManifest::new(),
            true,
            &ScanSettings::default(),
        );
        assert!(filtered
            .chunks
            .iter()
            .any(|c| c.doc_rel_path.starts_with("src/")));
        assert!(!filtered
            .chunks
            .iter()
            .any(|c| c.doc_rel_path.starts_with("tmp/")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn preferred_local_repo_binary_picks_newest_build() {
        let root = temp_dir("preferred-bin");
        fs::create_dir_all(root.join("crates").join("retrivio")).expect("mk crate dir");
        fs::write(root.join("Cargo.toml"), "[workspace]\nmembers=[]\n").expect("write cargo");
        let exe_name = current_platform_exe_name();
        let release = root.join("target").join("release").join(&exe_name);
        let debug = root.join("target").join("debug").join(&exe_name);
        write_executable(&release);
        thread::sleep(Duration::from_millis(20));
        write_executable(&debug);

        let preferred = preferred_local_repo_binary(&root).expect("preferred binary");
        assert_eq!(preferred, debug);
        assert_eq!(
            repo_root_from_anchor_path(&debug).expect("repo from binary"),
            root
        );

        let _ = fs::remove_dir_all(&root);
    }
}

pub(crate) fn has_indexable_files_in_root(root: &Path, settings: &ScanSettings) -> bool {
    let Ok(rd) = fs::read_dir(root) else {
        return false;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if !ft.is_file() {
            continue;
        }
        let lname = name.to_lowercase();
        if lname == "readme" || lname == "readme.md" || lname == "notes.txt" {
            return true;
        }
        let ext = Path::new(&name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        if settings.is_indexable_suffix(&format!(".{}", ext)) {
            return true;
        }
    }
    false
}

/// Per-scan limits. The defaults are the historical constants; every one is a config key
/// (`max_files_per_project`, `max_chunks_per_project`, `max_chunks_per_file`,
/// `max_file_chars`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScanCaps {
    pub(crate) max_files_per_project: usize,
    pub(crate) max_chunks_per_project: usize,
    pub(crate) max_chunks_per_file: usize,
    /// Characters of a file's text the indexer looks at (the read cap).
    pub(crate) max_file_chars: usize,
    /// Documents (docx, pptx, ...) larger than this are refused without a read
    /// (`max_document_bytes`); a refusal counts as a failed document.
    pub(crate) max_document_bytes: u64,
    /// Declared uncompressed total an Office/OpenDocument archive may have
    /// (`max_document_uncompressed_bytes`); over it the document is refused unread.
    pub(crate) max_document_uncompressed_bytes: u64,
    /// Deadline for the PDF extraction child (`document_extract_timeout_ms`).
    pub(crate) document_extract_timeout_ms: u64,
}

impl Default for ScanCaps {
    fn default() -> Self {
        ScanCaps {
            max_files_per_project: 2000,
            max_chunks_per_project: 6000,
            max_chunks_per_file: 28,
            max_file_chars: 80_000,
            max_document_bytes: 200_000_000,
            max_document_uncompressed_bytes: documents::DEFAULT_MAX_DOCUMENT_UNCOMPRESSED_BYTES,
            document_extract_timeout_ms: documents::DEFAULT_DOCUMENT_EXTRACT_TIMEOUT_MS,
        }
    }
}

impl ScanCaps {
    /// `files|chunks|chunks_per_file|chars|documents|document_bytes|uncompressed_bytes`:
    /// stored in `app_state` after a complete run; a different value on the next run revisits
    /// every project. The extraction timeout is a timing parameter, not part of it.
    pub(crate) fn fingerprint(&self, settings: &ScanSettings) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}",
            self.max_files_per_project,
            self.max_chunks_per_project,
            self.max_chunks_per_file,
            self.max_file_chars,
            settings.index_documents,
            self.max_document_bytes,
            self.max_document_uncompressed_bytes
        )
    }

    pub(crate) fn from_cfg(cfg: &ConfigValues) -> Self {
        ScanCaps {
            max_files_per_project: cfg.max_files_per_project.max(1) as usize,
            max_chunks_per_project: cfg.max_chunks_per_project.max(1) as usize,
            max_chunks_per_file: cfg.max_chunks_per_file.max(1) as usize,
            max_file_chars: cfg.max_file_chars.max(1) as usize,
            max_document_bytes: cfg.max_document_bytes.max(1) as u64,
            max_document_uncompressed_bytes: cfg.max_document_uncompressed_bytes.max(1) as u64,
            document_extract_timeout_ms: cfg.document_extract_timeout_ms.max(1) as u64,
        }
    }

    /// The bounds one document extraction gets: the text cut at about `max_file_chars`
    /// characters (the collector cuts at exactly that many afterwards), the archive and PDF
    /// bounds from the config keys.
    fn extract_limits(&self) -> documents::ExtractLimits {
        documents::ExtractLimits {
            archive_uncompressed_bytes: self.max_document_uncompressed_bytes,
            pdf_timeout: Duration::from_millis(self.document_extract_timeout_ms),
            ..documents::ExtractLimits::with_text_bytes(
                self.max_file_chars.saturating_mul(4).saturating_add(16),
            )
        }
    }
}

/// Files larger than this are never indexed. A byte bound on what is read at all, distinct
/// from the `max_file_chars` text cap; not configurable.
pub(crate) const MAX_FILE_BYTES: u64 = 2_000_000;

/// One indexable file the walk found, before selection.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CandidateFile {
    pub(crate) rel_path: String,
    /// Where to read it: the project directory joined with `rel_path` (follows symlinks).
    pub(crate) fs_path: PathBuf,
    pub(crate) size: u64,
    pub(crate) mtime: f64,
    /// Modification time in whole nanoseconds, the form the signature hashes.
    pub(crate) mtime_ns: i128,
    /// Selection tier (see [`selection_tier`]): 1 human documents, 2 code, 3 config and data.
    pub(crate) tier: u8,
}

/// What one walk of a project directory found.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ProjectListing {
    /// Indexable files within the size bounds, in directory order (unsorted).
    pub(crate) candidates: Vec<CandidateFile>,
    /// Every regular file's relative path (any suffix), for the summary's file list.
    pub(crate) file_names: Vec<String>,
    /// Newest mtime seen, the project directory itself included.
    pub(crate) latest_mtime: f64,
    /// Directory entries the walk could not read (`read_dir`, entry, file type or metadata
    /// errors). Non-zero means the listing is incomplete: whatever is missing must not be
    /// treated as deleted.
    pub(crate) unreadable: usize,
}

/// Walk a project directory once. Shared by the gate ([`project_scan`]) and the collector, so
/// the signature the gate stores describes exactly the files the collector indexed. With
/// `shallow`, only the files directly in `project_dir` are listed (a root-files project).
///
/// The project directory is canonicalised once; every entry's absolute path is the canonical
/// project path joined with its relative path (the walk never descends into symlinked
/// directories, so the two agree), and the exclude set is matched against that lexically.
pub(crate) fn walk_project_files(
    project_dir: &Path,
    exclude_abs: &HashSet<PathBuf>,
    shallow: bool,
    settings: &ScanSettings,
) -> ProjectListing {
    let mut out = ProjectListing {
        latest_mtime: file_mtime(project_dir).unwrap_or(0.0),
        ..ProjectListing::default()
    };
    let project_path = normalize_path(&project_dir.to_string_lossy());
    let mut stack = vec![project_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => {
                out.unreadable += 1;
                continue;
            }
        };
        for entry in rd {
            let Ok(entry) = entry else {
                out.unreadable += 1;
                continue;
            };
            let path = entry.path();
            let rel = path
                .strip_prefix(project_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            if !exclude_abs.is_empty() && path_is_under_any(&project_path.join(&rel), exclude_abs) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(ft) = entry.file_type() else {
                out.unreadable += 1;
                continue;
            };
            if ft.is_dir() {
                if shallow || name.starts_with('.') || settings.is_skip_dir(&name) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            // Regular files only: symlinks (to files or directories) and special files are
            // skipped, as they always were. Office owner/lock files (`~$deck.pptx`, a few
            // bytes naming who has the document open) are not documents.
            if !ft.is_file() || name.starts_with('.') || name.starts_with("~$") {
                continue;
            }
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => {
                    out.unreadable += 1;
                    continue;
                }
            };
            out.file_names.push(rel.clone());
            let mtime = metadata_mtime(&meta);
            if mtime > out.latest_mtime {
                out.latest_mtime = mtime;
            }
            let suffix = format!(
                ".{}",
                path.extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_lowercase()
            );
            if !settings.is_indexable_suffix(&suffix) {
                continue;
            }
            // Text files have a fixed byte bound; document formats are bounded by
            // `max_document_bytes` in the collector, where the refusal is counted.
            if meta.len() == 0
                || (meta.len() > MAX_FILE_BYTES && !documents::is_document_only_suffix(&suffix))
            {
                continue;
            }
            out.candidates.push(CandidateFile {
                rel_path: rel,
                tier: selection_tier(&suffix),
                fs_path: path,
                size: meta.len(),
                mtime,
                mtime_ns: metadata_mtime_ns(&meta),
            });
        }
    }
    out.file_names.sort();
    out
}

/// The files a scan indexes, in a fully deterministic order: by tier (human documents, then
/// code, then config and data; see [`selection_tier`]), newest first within a tier, then by
/// relative path, cut at `max_files_per_project`. The collector consumes the same order, so
/// when `max_chunks_per_project` bites, config and data files are the first left out and
/// notes the last. The same input always yields the same selection, so nothing shuffles in or
/// out between runs unless a file changes. Returns the selection and how many candidates the
/// file cap left out.
pub(crate) fn select_scan_files(
    listing: &ProjectListing,
    caps: &ScanCaps,
) -> (Vec<CandidateFile>, usize) {
    let mut files = listing.candidates.clone();
    files.sort_by(|a, b| {
        a.tier
            .cmp(&b.tier)
            .then_with(|| b.mtime.total_cmp(&a.mtime))
            .then_with(|| a.rel_path.cmp(&b.rel_path))
    });
    let evicted = files.len().saturating_sub(caps.max_files_per_project);
    files.truncate(caps.max_files_per_project);
    (files, evicted)
}

/// `count:xxh64` over the sorted `(rel_path, size, mtime_ns)` tuples of the selected files.
/// It changes when a selected file is added, removed, renamed, resized or touched (an edit
/// with a preserved timestamp still changes the size in almost every case; a future-dated
/// file cannot mask a later edit to another file); it does not change for files the scan
/// would not index anyway. This is the project gate's whole memory of the file system.
pub(crate) fn scan_signature_for(selected: &[CandidateFile]) -> String {
    let mut rows: Vec<(&str, u64, i128)> = selected
        .iter()
        .map(|f| (f.rel_path.as_str(), f.size, f.mtime_ns))
        .collect();
    rows.sort();
    let mut buf = String::new();
    for (rel, size, ns) in &rows {
        buf.push_str(rel);
        buf.push('\0');
        buf.push_str(&size.to_string());
        buf.push('\0');
        buf.push_str(&ns.to_string());
        buf.push('\n');
    }
    format!("{}:{}", rows.len(), content_hash_xxh64(buf.as_bytes()))
}

/// The gate's view of a project: one walk, the deterministic selection and its signature.
/// Handed to [`collect_project_corpus`] so the collector never walks again.
#[derive(Clone, Debug)]
pub(crate) struct ProjectScan {
    pub(crate) listing: ProjectListing,
    pub(crate) selected: Vec<CandidateFile>,
    /// Candidates `max_files_per_project` left out.
    evicted_by_file_cap: usize,
    pub(crate) signature: String,
    /// A root-files project: the root's direct files only.
    pub(crate) shallow: bool,
}

impl ProjectScan {
    pub(crate) fn latest_mtime(&self) -> f64 {
        self.listing.latest_mtime
    }

    /// True when every directory entry could be read.
    pub(crate) fn complete(&self) -> bool {
        self.listing.unreadable == 0
    }
}

pub(crate) fn project_scan(
    project_dir: &Path,
    exclude_abs: &HashSet<PathBuf>,
    caps: &ScanCaps,
    shallow: bool,
    settings: &ScanSettings,
) -> ProjectScan {
    let listing = walk_project_files(project_dir, exclude_abs, shallow, settings);
    let (selected, evicted_by_file_cap) = select_scan_files(&listing, caps);
    let signature = scan_signature_for(&selected);
    ProjectScan {
        listing,
        selected,
        evicted_by_file_cap,
        signature,
        shallow,
    }
}

/// Produce the project's chunks for this run from a finished [`ProjectScan`].
///
/// Every selected file is listed in the returned `files` (the full keep set); only files that
/// are new or changed against `manifest` (or all of them with `rechunk_all`) are read,
/// chunked and listed in `chunks` (the work set). Unchanged files contribute their manifest
/// chunk count towards the per-project cap and are kept whole by the prune step. Change
/// detection is the manifest's size+mtime fast path, then the content hash when the stat
/// differs (touch without edit; such files get their new stat written, nothing else).
///
/// `doc_path` is `<canonical project path>/<rel_path>` for every file and chunk, so the keep
/// set and the stored rows are keyed the same way regardless of symlinks.
pub(crate) fn collect_project_corpus(
    project_dir: &Path,
    scan: &ProjectScan,
    caps: &ScanCaps,
    max_chars: usize,
    manifest: &FileManifest,
    rechunk_all: bool,
    settings: &ScanSettings,
) -> ProjectCorpus {
    collect_project_corpus_verifying(
        project_dir,
        scan,
        caps,
        max_chars,
        manifest,
        rechunk_all,
        &HashSet::new(),
        settings,
    )
}

/// The relative paths of `verify_paths` (absolute, normalised) that lie inside `project`.
pub(crate) fn verify_rel_paths(project: &Path, verify_paths: &[PathBuf]) -> HashSet<String> {
    verify_paths
        .iter()
        .filter_map(|p| p.strip_prefix(project).ok())
        .filter(|rel| !rel.as_os_str().is_empty())
        .map(|rel| rel.to_string_lossy().to_string())
        .collect()
}

/// A verified file: its content hash against the manifest whatever its stat says.
/// Returns (changed, content_hash).
pub(crate) fn file_content_changed(
    manifest: &FileManifest,
    rel_path: &str,
    content: &[u8],
) -> (bool, String) {
    let hash = content_hash_xxh64(content);
    match manifest.get(rel_path) {
        Some(entry) if entry.content_hash == hash => (false, hash),
        _ => (true, hash),
    }
}

/// [`collect_project_corpus`] with `verify`: relative paths that skip the size+mtime fast path
/// and are read and hashed regardless (see [`run_native_index_verifying`]).
#[allow(clippy::too_many_arguments)] // the collector's inputs: scan, caps, settings, manifest and the per-run switches
pub(crate) fn collect_project_corpus_verifying(
    project_dir: &Path,
    scan: &ProjectScan,
    caps: &ScanCaps,
    max_chars: usize,
    manifest: &FileManifest,
    rechunk_all: bool,
    verify: &HashSet<String>,
    settings: &ScanSettings,
) -> ProjectCorpus {
    const CHUNK_SIZE_CHARS: usize = 1000;
    const CHUNK_OVERLAP_CHARS: usize = 180;
    const SUMMARY_SNIPPET_CHARS: usize = 900;
    const SUMMARY_SNIPPET_FILES: usize = 30;
    const AST_CHUNK_SIZE: usize = 1500; // larger for AST chunks since they're semantic units

    #[cfg(test)]
    {
        let target = INJECT_COLLECTOR_PANIC
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        if let Some(name) = target {
            if project_dir.file_name().and_then(|s| s.to_str()) == Some(name.as_str()) {
                panic!("injected collector panic for {}", name);
            }
        }
    }

    let project_path = normalize_path(&project_dir.to_string_lossy());
    let doc_path_for =
        |rel: &str| -> String { project_path.join(rel).to_string_lossy().to_string() };

    let mut chunks: Vec<ProjectChunk> = Vec::new();
    let mut files: Vec<ScannedFile> = Vec::new();
    // The summary quotes the first SUMMARY_SNIPPET_FILES selected files in relative-path
    // order (not selection order, which follows mtime), so the summary text is independent
    // of touches and the project vector is reused until content or membership changes.
    // Documents are never quoted: their raw bytes are not text, and extracting an unchanged
    // document on every scan is exactly what the manifest fast path avoids.
    let is_document_candidate = |cand: &CandidateFile| -> bool {
        settings.is_document_suffix(&suffix_with_dot(&cand.fs_path))
    };
    let snippet_files: HashSet<&str> = {
        let mut rels: Vec<&str> = scan
            .selected
            .iter()
            .filter(|c| !is_document_candidate(c))
            .map(|c| c.rel_path.as_str())
            .collect();
        rels.sort_unstable();
        rels.into_iter().take(SUMMARY_SNIPPET_FILES).collect()
    };
    let mut snippets: BTreeMap<String, String> = BTreeMap::new();
    let mut documents_extracted = 0usize;
    let mut document_failures: Vec<(String, String)> = Vec::new();
    // Chunks of unchanged files count towards the per-project cap without being re-produced.
    let mut carried_chunks = 0usize;
    let mut visited = 0usize;
    let mut files_truncated = 0usize;
    // Selected files whose read failed after the walk listed them: the scan is incomplete
    // (prune and the signature update are skipped), they are not treated as deleted.
    let mut read_failures = 0usize;
    let mut chunk_cap_hit = false;
    let mut file_chunk_cap_hit = false;
    let mut text_cap_hit = false;
    let extract_limits = caps.extract_limits();

    // A document that fails extraction (refused by a bound, corrupt, parser error or panic,
    // PDF child timed out or killed) must never cost content that was indexed before: when
    // the manifest knows the file, it is carried exactly as an unchanged file (its stored
    // chunks, vectors, symbols and manifest row stand; the manifest entry is not advanced,
    // so the file is retried on the next full revisit), and the failure is reported once for
    // this run. A never-indexed document that fails is simply absent from the keep set.
    let carry_failed_document =
        |rel: String,
         doc_path: String,
         reason: String,
         files: &mut Vec<ScannedFile>,
         carried: &mut usize,
         failures: &mut Vec<(String, String)>| {
            failures.push((rel.clone(), reason));
            if let Some(entry) = manifest.get(&rel) {
                *carried += entry.chunk_count.max(0) as usize;
                files.push(ScannedFile {
                    rel_path: rel,
                    doc_path,
                    size: entry.size,
                    mtime: entry.mtime,
                    content_hash: entry.content_hash.clone(),
                    chunk_count: entry.chunk_count,
                    rechunked: false,
                    stat_changed: false,
                });
            }
        };

    for cand in &scan.selected {
        if carried_chunks + chunks.len() >= caps.max_chunks_per_project {
            chunk_cap_hit = true;
            break;
        }
        visited += 1;
        let rel = cand.rel_path.clone();
        let doc_path = doc_path_for(&rel);
        let size_i64 = cand.size as i64;
        let doc_mtime = cand.mtime;

        // Fast path: manifest says size and mtime are what they were. No read, no chunking;
        // the first few files are still read for the project summary snippet. A file named
        // for verification never takes it.
        let verify_this = verify.contains(rel.as_str());
        if !rechunk_all && !verify_this {
            if let Some(entry) = manifest_stat_match(manifest, &rel, size_i64, doc_mtime) {
                if snippet_files.contains(rel.as_str()) {
                    if let Some((text, _)) = read_for_index(&cand.fs_path, caps.max_file_chars) {
                        let snippet: String = text.chars().take(SUMMARY_SNIPPET_CHARS).collect();
                        snippets.insert(rel.clone(), format!("{}\n{}", rel, snippet));
                    }
                }
                carried_chunks += entry.chunk_count.max(0) as usize;
                files.push(ScannedFile {
                    rel_path: rel,
                    doc_path,
                    size: size_i64,
                    mtime: doc_mtime,
                    content_hash: entry.content_hash.clone(),
                    chunk_count: entry.chunk_count,
                    rechunked: false,
                    stat_changed: false,
                });
                continue;
            }
        }

        let is_document = is_document_candidate(cand);
        if is_document && cand.size > caps.max_document_bytes {
            // Refused without a read; a previously indexed version stays as it was.
            carry_failed_document(
                rel,
                doc_path,
                format!(
                    "{} bytes exceeds max_document_bytes={}",
                    cand.size, caps.max_document_bytes
                ),
                &mut files,
                &mut carried_chunks,
                &mut document_failures,
            );
            continue;
        }

        let raw = match fs::read(&cand.fs_path) {
            Ok(raw) => raw,
            Err(_) => {
                read_failures += 1;
                continue;
            }
        };

        // Stat differed (or the file is new, or it is up for verification): the content hash
        // decides.
        let (changed, content_hash) = if rechunk_all {
            (true, content_hash_xxh64(&raw))
        } else if verify_this {
            file_content_changed(manifest, &rel, &raw)
        } else {
            file_has_changed(manifest, &rel, size_i64, doc_mtime, &raw)
        };
        if !changed {
            let entry = manifest.get(&rel);
            let chunk_count = entry.map(|e| e.chunk_count).unwrap_or(0);
            carried_chunks += chunk_count.max(0) as usize;
            files.push(ScannedFile {
                rel_path: rel.clone(),
                doc_path,
                size: size_i64,
                mtime: doc_mtime,
                content_hash,
                chunk_count,
                rechunked: false,
                // A verified file whose stat did not move needs no manifest refresh.
                stat_changed: manifest_stat_match(manifest, &rel, size_i64, doc_mtime).is_none(),
            });
            continue;
        }

        // Documents go through the extractor (changed files only: an unchanged document was
        // handled by the manifest above and is never re-extracted). A failure keeps the
        // previously indexed version of the file, if any (see `carry_failed_document`).
        let mut document_cut = false;
        let indexed_text = if is_document {
            match documents::extract_from_bytes(&cand.fs_path, &raw, &extract_limits) {
                Ok(Some(doc)) => {
                    documents_extracted += 1;
                    document_cut = doc.truncated;
                    let body = match doc.title {
                        Some(title) if !doc.text.starts_with(&title) => {
                            format!("{}\n{}", title, doc.text)
                        }
                        _ => doc.text,
                    };
                    cap_indexed_text(&collapse_whitespace(&body), caps.max_file_chars)
                }
                Ok(None) => index_text_from_bytes(&cand.fs_path, &raw, caps.max_file_chars),
                Err(reason) => {
                    carry_failed_document(
                        rel,
                        doc_path,
                        reason,
                        &mut files,
                        &mut carried_chunks,
                        &mut document_failures,
                    );
                    continue;
                }
            }
        } else {
            index_text_from_bytes(&cand.fs_path, &raw, caps.max_file_chars)
        };
        let Some((text, text_truncated)) = indexed_text else {
            // Selected, readable, but without indexable text (whitespace only, or a document
            // that yielded none): a file with zero chunks. It stays in the keep set, so its
            // manifest row is written and its stored chunks (from an earlier, non-empty
            // version) are pruned; a code file's symbol and import rows are replaced by the
            // empty extraction.
            files.push(ScannedFile {
                rel_path: rel,
                doc_path,
                size: size_i64,
                mtime: doc_mtime,
                content_hash,
                chunk_count: 0,
                rechunked: true,
                stat_changed: false,
            });
            continue;
        };
        if snippet_files.contains(rel.as_str()) {
            let snippet: String = text.chars().take(SUMMARY_SNIPPET_CHARS).collect();
            snippets.insert(rel.clone(), format!("{}\n{}", rel, snippet));
        }

        // Try AST-aware chunking for code files, fall back to text windows for prose
        let is_code = code_intel::language_for_extension(
            &cand
                .fs_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase(),
        )
        .is_some();

        let produced_before = chunks.len();
        let mut truncated = text_truncated || document_cut;
        if truncated {
            text_cap_hit = true;
        }
        if is_code {
            // AST-aware chunking: produces semantic chunks (functions, classes, imports)
            let semantic_chunks =
                code_intel::analyze_file(&cand.fs_path, &text, &rel, AST_CHUNK_SIZE);
            for (chunk_index, sc) in semantic_chunks.into_iter().enumerate() {
                if carried_chunks + chunks.len() >= caps.max_chunks_per_project {
                    chunk_cap_hit = true;
                    truncated = true;
                    break;
                }
                if chunk_index >= caps.max_chunks_per_file {
                    file_chunk_cap_hit = true;
                    truncated = true;
                    break;
                }
                let token_count = word_tokens(&sc.text).len() as i64;
                if token_count == 0 {
                    continue;
                }
                let mut hasher = Sha1::new();
                hasher.update(sc.text.as_bytes());
                let text_hash = format!("{:x}", hasher.finalize());
                let header = code_intel::build_context_header(&sc, &rel);
                let kind_str = match sc.kind {
                    code_intel::ChunkKind::ImportBlock => "import_block",
                    code_intel::ChunkKind::Preamble => "preamble",
                    code_intel::ChunkKind::Function => "function",
                    code_intel::ChunkKind::TypeHeader => "type_header",
                    code_intel::ChunkKind::Method => "method",
                    code_intel::ChunkKind::Declaration => "declaration",
                    code_intel::ChunkKind::TextWindow => "text_window",
                };
                chunks.push(ProjectChunk {
                    doc_path: doc_path.clone(),
                    doc_rel_path: rel.clone(),
                    doc_mtime,
                    chunk_index: chunk_index as i64,
                    token_count,
                    text_hash,
                    text: sc.text,
                    chunk_kind: kind_str.to_string(),
                    symbol_name: sc.symbol_name,
                    parent_context: sc.parent_context,
                    line_start: sc.line_start as i64,
                    line_end: sc.line_end as i64,
                    context_header: header,
                });
            }
        } else {
            // Fallback: character-window chunking for prose/non-code files. `chunk_text`
            // stops at the per-file cap itself; ask for one more to learn whether it did.
            let windows = chunk_text(
                &text,
                CHUNK_SIZE_CHARS,
                CHUNK_OVERLAP_CHARS,
                caps.max_chunks_per_file + 1,
            );
            if windows.len() > caps.max_chunks_per_file {
                file_chunk_cap_hit = true;
                truncated = true;
            }
            for (chunk_index, chunk_text) in windows
                .into_iter()
                .take(caps.max_chunks_per_file)
                .enumerate()
            {
                if carried_chunks + chunks.len() >= caps.max_chunks_per_project {
                    chunk_cap_hit = true;
                    truncated = true;
                    break;
                }
                let token_count = word_tokens(&chunk_text).len() as i64;
                if token_count == 0 {
                    continue;
                }
                let mut hasher = Sha1::new();
                hasher.update(chunk_text.as_bytes());
                let text_hash = format!("{:x}", hasher.finalize());
                chunks.push(ProjectChunk {
                    doc_path: doc_path.clone(),
                    doc_rel_path: rel.clone(),
                    doc_mtime,
                    chunk_index: chunk_index as i64,
                    token_count,
                    text_hash,
                    text: chunk_text,
                    chunk_kind: "text_window".to_string(),
                    symbol_name: String::new(),
                    parent_context: String::new(),
                    line_start: 0,
                    line_end: 0,
                    context_header: String::new(),
                });
            }
        }
        if truncated {
            files_truncated += 1;
        }
        files.push(ScannedFile {
            rel_path: rel,
            doc_path,
            size: size_i64,
            mtime: doc_mtime,
            content_hash,
            chunk_count: (chunks.len() - produced_before) as i64,
            rechunked: true,
            stat_changed: false,
        });
    }
    // Selected files the project chunk cap kept the loop from reaching at all.
    let evicted_by_chunk_cap = scan.selected.len().saturating_sub(visited);

    let mut caps_hit: Vec<String> = Vec::new();
    if scan.evicted_by_file_cap > 0 {
        caps_hit.push(format!(
            "{} files not indexed (max_files_per_project={})",
            scan.evicted_by_file_cap, caps.max_files_per_project
        ));
    }
    if evicted_by_chunk_cap > 0 || chunk_cap_hit {
        caps_hit.push(format!(
            "{} files not indexed (max_chunks_per_project={})",
            evicted_by_chunk_cap, caps.max_chunks_per_project
        ));
    }
    if file_chunk_cap_hit {
        caps_hit.push(format!(
            "chunks cut by max_chunks_per_file={}",
            caps.max_chunks_per_file
        ));
    }
    if text_cap_hit {
        caps_hit.push(format!(
            "text cut by max_file_chars={}",
            caps.max_file_chars
        ));
    }

    // The summary's file list is sorted (the walk is), so the summary text is byte-identical
    // between runs when nothing changed and the project vector is reused. `indexed_files` is
    // the number of selected files holding at least one chunk, computed the same way for
    // re-chunked, touched and untouched files, so a touch or a zero-chunk file never moves it.
    let indexed_files = files.iter().filter(|f| f.chunk_count > 0).count();
    let names_section = scan
        .listing
        .file_names
        .iter()
        .take(500)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let snippet_section = snippets.into_values().collect::<Vec<_>>().join("\n\n");
    let base_name = project_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("project")
        .to_string();
    let mut title = base_name.replace(['-', '_'], " ");
    if scan.shallow {
        title = format!("{} (root files)", title);
    }
    let mut summary = format!(
        "project {}{}\nindexed_files {}\nfiles {}\n\n{}",
        base_name,
        if scan.shallow { " (root files)" } else { "" },
        indexed_files,
        names_section,
        snippet_section
    );
    if summary.chars().count() > max_chars {
        summary = summary.chars().take(max_chars).collect();
    }

    ProjectCorpus {
        doc: ProjectDoc {
            path: project_path,
            title,
            summary,
            mtime: scan.latest_mtime(),
        },
        chunks,
        files,
        scan_signature: scan.signature.clone(),
        complete: scan.complete() && read_failures == 0,
        files_unreadable: (scan.listing.unreadable + read_failures) as i64,
        files_evicted_by_cap: (scan.evicted_by_file_cap + evicted_by_chunk_cap) as i64,
        files_truncated_by_cap: files_truncated as i64,
        caps_note: caps_hit.join("; "),
        documents_extracted: documents_extracted as i64,
        documents_failed: document_failures.len() as i64,
        document_failures,
    }
}

/// `.docx` for `a/b.DOCX`; empty when the path has no extension.
pub(crate) fn suffix_with_dot(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_ascii_lowercase()))
        .unwrap_or_default()
}

/// Read a file the way the indexer does (see [`index_text_from_bytes`]); `None` when it is
/// unreadable or has no indexable text. The flag is true when the text was cut at `max_chars`.
pub(crate) fn read_for_index(path: &Path, max_chars: usize) -> Option<(String, bool)> {
    let raw = fs::read(path).ok()?;
    index_text_from_bytes(path, &raw, max_chars)
}

/// The text the indexer works on for a file's raw bytes: whitespace is preserved for code
/// (indentation conveys scope) and collapsed for prose; empty results are `None`; cut at
/// `max_chars` characters (the `max_file_chars` cap), with the flag telling when that bit.
pub(crate) fn index_text_from_bytes(
    path: &Path,
    raw: &[u8],
    max_chars: usize,
) -> Option<(String, bool)> {
    let text = String::from_utf8_lossy(raw);
    let is_code = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| code_intel::language_for_extension(&ext.to_lowercase()).is_some())
        .unwrap_or(false);
    let cleaned = if is_code {
        text.to_string()
    } else {
        collapse_whitespace(&text)
    };
    cap_indexed_text(&cleaned, max_chars)
}

/// Cut already-cleaned text at `max_chars` characters; `None` when it is blank. The flag is
/// true when something was cut.
pub(crate) fn cap_indexed_text(cleaned: &str, max_chars: usize) -> Option<(String, bool)> {
    if cleaned.trim().is_empty() {
        return None;
    }
    let mut out = String::with_capacity(cleaned.len().min(max_chars.saturating_mul(4)));
    let mut iter = cleaned.chars();
    for ch in iter.by_ref().take(max_chars) {
        out.push(ch);
    }
    let truncated = iter.next().is_some();
    Some((out, truncated))
}

/// Compute xxhash64 of file contents. Very fast (~2GB/s).
pub(crate) fn content_hash_xxh64(content: &[u8]) -> String {
    format!("{:016x}", xxh64::xxh64(content, 0))
}

/// One row of the per-project file manifest (`project_files`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FileManifestEntry {
    pub(crate) size: i64,
    pub(crate) mtime: f64,
    pub(crate) content_hash: String,
    pub(crate) chunk_count: i64,
}

/// rel_path -> manifest entry for one project.
pub(crate) type FileManifest = HashMap<String, FileManifestEntry>;

/// Load the existing file manifest for a project from SQLite.
pub(crate) fn load_file_manifest(
    conn: &Connection,
    project_id: i64,
) -> Result<FileManifest, String> {
    let mut stmt = conn
        .prepare("SELECT rel_path, file_size, file_mtime, content_hash, chunk_count FROM project_files WHERE project_id = ?1")
        .map_err(|e| format!("failed preparing file manifest query: {}", e))?;
    let rows = stmt
        .query_map(params![project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                FileManifestEntry {
                    size: row.get::<_, i64>(1)?,
                    mtime: row.get::<_, f64>(2)?,
                    content_hash: row.get::<_, String>(3)?,
                    chunk_count: row.get::<_, i64>(4)?,
                },
            ))
        })
        .map_err(|e| format!("failed querying file manifest: {}", e))?;
    let mut out = HashMap::new();
    for row in rows {
        let (rel, entry) = row.map_err(|e| format!("failed reading file manifest row: {}", e))?;
        out.insert(rel, entry);
    }
    Ok(out)
}

/// Update the file manifest entry after indexing a file.
#[allow(clippy::too_many_arguments)] // one manifest row, one argument per column
pub(crate) fn upsert_file_manifest(
    conn: &Connection,
    project_id: i64,
    rel_path: &str,
    abs_path: &str,
    file_size: i64,
    file_mtime: f64,
    content_hash: &str,
    chunk_count: i64,
) -> Result<(), String> {
    let now = now_ts();
    conn.execute(
        r#"
INSERT INTO project_files(project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, last_indexed)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT(project_id, rel_path) DO UPDATE SET
    abs_path = excluded.abs_path,
    file_size = excluded.file_size,
    file_mtime = excluded.file_mtime,
    content_hash = excluded.content_hash,
    chunk_count = excluded.chunk_count,
    last_indexed = excluded.last_indexed
"#,
        params![project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, now],
    )
    .map_err(|e| format!("failed upserting file manifest: {}", e))?;
    Ok(())
}

/// The manifest fast path: the entry for `rel_path` when its recorded size and mtime equal the
/// file's current ones, so the file can be treated as unchanged without reading it.
pub(crate) fn manifest_stat_match<'a>(
    manifest: &'a FileManifest,
    rel_path: &str,
    file_size: i64,
    file_mtime: f64,
) -> Option<&'a FileManifestEntry> {
    manifest
        .get(rel_path)
        .filter(|e| e.size == file_size && (e.mtime - file_mtime).abs() < 0.001)
}

/// Check if a file has changed by comparing mtime+size, then verifying with content hash.
/// Returns (changed: bool, content_hash: String).
pub(crate) fn file_has_changed(
    manifest: &FileManifest,
    rel_path: &str,
    file_size: i64,
    file_mtime: f64,
    content: &[u8],
) -> (bool, String) {
    if let Some(entry) = manifest_stat_match(manifest, rel_path, file_size, file_mtime) {
        return (false, entry.content_hash.clone());
    }
    let hash = content_hash_xxh64(content);
    match manifest.get(rel_path) {
        None => (true, hash), // New file
        Some(entry) => {
            // Stat differs but the content is the same (touch, clock skew, restored copy).
            if entry.content_hash == hash {
                return (false, hash);
            }
            (true, hash)
        }
    }
}

pub(crate) fn chunk_text(
    text: &str,
    size: usize,
    overlap: usize,
    max_chunks: usize,
) -> Vec<String> {
    let cleaned = collapse_whitespace(text);
    if cleaned.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = cleaned.chars().collect();
    if chars.len() <= size {
        return vec![cleaned];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    let step = size.saturating_sub(overlap).max(1);
    let n = chars.len();
    while start < n && out.len() < max_chunks {
        let mut end = (start + size).min(n);
        let mut window: String = chars[start..end].iter().collect();
        if end < n {
            if let Some(split) = window.rfind(' ') {
                if split > ((size as f32) * 0.60) as usize {
                    window = window[..split].to_string();
                    end = start + window.chars().count();
                }
            }
        }
        let trimmed = window.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
        if end >= n {
            break;
        }
        start += step;
    }
    out
}
