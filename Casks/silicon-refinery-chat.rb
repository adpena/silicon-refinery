cask "silicon-refinery-chat" do
  version "0.1.0"
  sha256 "48f71223bd4c3f52e5b8d2ca52c771346f0da5ab1906b5b0d1dd10ddad798945"

  url "https://github.com/adpena/silicon-refinery-chat/releases/download/v#{version}/SiliconRefineryChat-#{version}.dmg"
  name "SiliconRefineryChat"
  desc "Standalone macOS app for local Apple Foundation Models chat"
  homepage "https://github.com/adpena/silicon-refinery-chat"

  app "SiliconRefineryChat.app"

  zap trash: [
    "~/Library/Application Support/com.siliconrefinery.chat",
    "~/Library/Preferences/com.siliconrefinery.chat.plist",
  ]
end
