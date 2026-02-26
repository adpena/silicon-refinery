cask "silicon-refinery-chat" do
  version "0.0.213"
  sha256 "ede289a25d65c266fcc72e3c1e1e6a28d52464bfdf8ae9c2042bbb8045fd5449"

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
