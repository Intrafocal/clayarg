// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ClayargCapture",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "clayarg-capture",
            path: "Sources/ClayargCapture"
        )
    ]
)
