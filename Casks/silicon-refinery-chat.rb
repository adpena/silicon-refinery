cask "silicon-refinery-chat" do
  version "0.0.209"
  sha256 "ffd879f3841b3ba4d1a2def6536ba02206dc0113776436661cd2a8955665bd32"

  url "https://github.com/adpena/silicon-refinery-chat/releases/download/v#{version}/SiliconRefineryChat-#{version}.dmg"
  name "SiliconRefineryChat"
  desc "Standalone macOS app for local Apple Foundation Models chat"
  homepage "https://github.com/adpena/silicon-refinery-chat"

  app "SiliconRefineryChat.app"
  binary "#{appdir}/SiliconRefineryChat.app/Contents/MacOS/SiliconRefineryChat",
         target: "silicon-refinery-chat"

  zap trash: [
    "~/Library/Application Support/com.siliconrefinery.chat",
    "~/Library/Preferences/com.siliconrefinery.chat.plist",
  ]
end
