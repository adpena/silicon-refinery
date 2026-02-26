class SiliconRefinery < Formula
  desc "Zero-trust local data refinery framework for Apple Foundation Models"
  homepage "https://github.com/adpena/silicon-refinery"
  url "https://github.com/adpena/silicon-refinery/archive/refs/tags/v0.0.217.tar.gz"
  sha256 "04dc5d79bdc9b47f12e1c09ff5b8514045a4a4760657415cb923bc5524eaeb88"
  license "MIT"
  head "https://github.com/adpena/silicon-refinery.git", branch: "main"

  depends_on "uv"

  def install
    libexec.install Dir["*"]

    (bin/"silicon-refinery").write <<~SH
      #!/bin/sh
      if [ ! -f "#{libexec}/README.md" ] && [ -f "#{prefix}/README.md" ]; then
        ln -sf "#{prefix}/README.md" "#{libexec}/README.md"
      fi
      exec "#{Formula["uv"].opt_bin}/uv" run --project "#{libexec}" --directory "#{libexec}" silicon-refinery "$@"
    SH
  end

  def caveats
    <<~EOS
      On first run, uv may resolve project dependencies (including apple-fm-sdk from GitHub).
      If your shell cannot find `silicon-refinery`, restart the terminal so Homebrew PATH updates apply.
    EOS
  end

  test do
    output = shell_output("#{bin}/silicon-refinery --help")
    assert_match "SiliconRefinery", output
  end
end
