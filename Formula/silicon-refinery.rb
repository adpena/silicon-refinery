class SiliconRefinery < Formula
  desc "Zero-trust local data refinery framework for Apple Foundation Models"
  homepage "https://github.com/adpena/silicon-refinery"
  license "MIT"

  # HEAD-only for now while Apple FM SDK dependency remains GitHub-sourced.
  head "https://github.com/adpena/silicon-refinery.git", branch: "main"

  depends_on "uv"

  def install
    libexec.install Dir["*"]

    (bin/"silicon-refinery").write <<~SH
      #!/bin/sh
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
