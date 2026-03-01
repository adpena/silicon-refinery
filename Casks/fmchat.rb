cask "fmchat" do
  version "0.0.217"
  sha256 "47307f78fe3971f7918ae49f5aa69d653acaefe868fff43287d448a17bb4db29"

  url "https://github.com/adpena/fmchat/releases/download/v#{version}/FMChat-#{version}.dmg"
  name "FMChat"
  desc "Standalone app for local Apple Foundation Models chat"
  homepage "https://github.com/adpena/fmchat"

  depends_on macos: ">= :big_sur"

  app "FMChat.app"
  binary "#{appdir}/FMChat.app/Contents/Resources/fmchat",
         target: "fmchat"

  zap trash: [
    "~/Library/Application Support/com.fmtools.chat",
    "~/Library/Preferences/com.fmtools.chat.plist",
  ]
end
