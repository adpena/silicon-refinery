cask "silicon-refinery-chat" do
  version "0.0.217"
  sha256 "47307f78fe3971f7918ae49f5aa69d653acaefe868fff43287d448a17bb4db29"

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
