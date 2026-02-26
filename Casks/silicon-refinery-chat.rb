cask "silicon-refinery-chat" do
  version "0.0.210"
  sha256 "586dce8a993dc1feba59cbae5ec4e2d2b129ca004b68c46a3b1d34fa7d5a83bb"

  url "https://github.com/adpena/silicon-refinery-chat/releases/download/v#{version}/SiliconRefineryChat-#{version}.dmg"
  name "SiliconRefineryChat"
  desc "Standalone app for local Apple Foundation Models chat"
  homepage "https://github.com/adpena/silicon-refinery-chat"

  depends_on macos: ">= :big_sur"

  app "SiliconRefineryChat.app"
  binary "#{appdir}/SiliconRefineryChat.app/Contents/Resources/silicon-refinery-chat",
         target: "silicon-refinery-chat"

  zap trash: [
    "~/Library/Application Support/com.siliconrefinery.chat",
    "~/Library/Preferences/com.siliconrefinery.chat.plist",
  ]
end
