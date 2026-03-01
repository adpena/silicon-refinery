class Fmtools < Formula
  desc "Async-native Python framework for Apple Foundation Models"
  homepage "https://github.com/adpena/fmtools"
  url "https://github.com/adpena/fmtools/archive/refs/tags/v0.0.217.tar.gz"
  sha256 "04dc5d79bdc9b47f12e1c09ff5b8514045a4a4760657415cb923bc5524eaeb88"
  license "MIT"
  head "https://github.com/adpena/fmtools.git", branch: "main"

  depends_on "uv"

  def install
    libexec.install Dir["*"]

    (bin/"fmtools").write <<~SH
      #!/bin/sh
      if [ ! -f "#{libexec}/README.md" ] && [ -f "#{prefix}/README.md" ]; then
        ln -sf "#{prefix}/README.md" "#{libexec}/README.md"
      fi
      exec "#{Formula["uv"].opt_bin}/uv" run --project "#{libexec}" --directory "#{libexec}" fmtools "$@"
    SH
  end

  def caveats
    <<~EOS
      On first run, uv may resolve project dependencies (including apple-fm-sdk from GitHub).
      If your shell cannot find `fmtools`, restart the terminal so Homebrew PATH updates apply.
    EOS
  end

  test do
    output = shell_output("#{bin}/fmtools --help")
    assert_match "FMTools", output
  end
end
