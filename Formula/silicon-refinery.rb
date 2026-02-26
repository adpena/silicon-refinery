class SiliconRefinery < Formula
  desc "Zero-trust local data refinery framework for Apple Foundation Models"
  homepage "https://github.com/adpena/silicon-refinery"
  url "https://github.com/adpena/silicon-refinery/archive/refs/tags/v0.0.214.tar.gz"
  sha256 "0e3af338ded127522b416ac93ff66b04f6a4fc39f5611470d35c0d9bb3d644b8"
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
