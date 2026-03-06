class Retrivio < Formula
  desc "Local-first semantic code intelligence and navigation index"
  homepage "https://github.com/stouffer-labs/retrivio"
  url "https://github.com/stouffer-labs/retrivio/archive/refs/heads/main.tar.gz"
  version "main"
  sha256 :no_check
  license "MIT"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args(path: "crates/retrivio")
  end

  test do
    assert_match "retrivio", shell_output("#{bin}/retrivio --help")
  end
end
