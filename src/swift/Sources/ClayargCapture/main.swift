import Foundation
import ModelIO
import RealityKit

// MARK: - JSON Progress Output

struct ProgressMessage: Encodable {
    let phase: String
    let progress: Double?
    let message: String?
    let output: String?
    let code: Int?
}

func emit(_ msg: ProgressMessage) {
    let encoder = JSONEncoder()
    if let data = try? encoder.encode(msg), let line = String(data: data, encoding: .utf8) {
        print(line)
        fflush(stdout)
    }
}

func emitProgress(_ fraction: Double, _ message: String) {
    emit(ProgressMessage(phase: "processing", progress: fraction, message: message, output: nil, code: nil))
}

func emitCompleted(_ outputPath: String) {
    emit(ProgressMessage(phase: "completed", progress: 1.0, message: nil, output: outputPath, code: nil))
}

func emitError(_ message: String, code: Int = 1) {
    emit(ProgressMessage(phase: "error", progress: nil, message: message, output: nil, code: code))
}

// MARK: - Argument Parsing

struct Config {
    let inputDir: URL
    let outputFile: URL
    let detail: PhotogrammetrySession.Request.Detail
    let featureSensitivity: PhotogrammetrySession.Configuration.FeatureSensitivity
    let emitProgress: Bool
}

func parseDetail(_ value: String) -> PhotogrammetrySession.Request.Detail? {
    switch value.lowercased() {
    case "preview": return .preview
    case "reduced": return .reduced
    case "medium": return .medium
    case "full": return .full
    case "raw": return .raw
    default: return nil
    }
}

func parseSensitivity(_ value: String) -> PhotogrammetrySession.Configuration.FeatureSensitivity? {
    switch value.lowercased() {
    case "normal": return .normal
    case "high": return .high
    default: return nil
    }
}

func parseArgs() -> Config? {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        fputs("Usage: clayarg-capture <input-dir> <output-file> [options]\n", stderr)
        fputs("  --detail <level>     preview|reduced|medium|full|raw (default: medium)\n", stderr)
        fputs("  --progress           Emit JSON progress lines to stdout\n", stderr)
        fputs("  --sensitivity <val>  Feature sensitivity: normal|high (default: normal)\n", stderr)
        return nil
    }

    let inputDir = URL(fileURLWithPath: args[1], isDirectory: true)
    let outputFile = URL(fileURLWithPath: args[2])

    var detail: PhotogrammetrySession.Request.Detail = .medium
    var sensitivity: PhotogrammetrySession.Configuration.FeatureSensitivity = .normal
    var progress = false

    var i = 3
    while i < args.count {
        switch args[i] {
        case "--detail":
            i += 1
            guard i < args.count, let d = parseDetail(args[i]) else {
                fputs("Error: Invalid detail level\n", stderr)
                return nil
            }
            detail = d
        case "--sensitivity":
            i += 1
            guard i < args.count, let s = parseSensitivity(args[i]) else {
                fputs("Error: Invalid sensitivity value\n", stderr)
                return nil
            }
            sensitivity = s
        case "--progress":
            progress = true
        default:
            fputs("Error: Unknown option '\(args[i])'\n", stderr)
            return nil
        }
        i += 1
    }

    return Config(
        inputDir: inputDir,
        outputFile: outputFile,
        detail: detail,
        featureSensitivity: sensitivity,
        emitProgress: progress
    )
}

// MARK: - Photogrammetry

func runCapture(config: Config) async throws {
    guard PhotogrammetrySession.isSupported else {
        emitError("Object Capture is not supported on this device", code: 2)
        exit(2)
    }

    var sessionConfig = PhotogrammetrySession.Configuration()
    sessionConfig.featureSensitivity = config.featureSensitivity

    let session: PhotogrammetrySession
    do {
        session = try PhotogrammetrySession(input: config.inputDir, configuration: sessionConfig)
    } catch {
        emitError("Failed to create session: \(error.localizedDescription)", code: 3)
        exit(3)
    }

    // Always output USDZ first (only format Object Capture supports)
    let usdzURL: URL
    let needsConversion = config.outputFile.pathExtension.lowercased() != "usdz"
    if needsConversion {
        usdzURL = config.outputFile.deletingPathExtension().appendingPathExtension("usdz")
    } else {
        usdzURL = config.outputFile
    }

    // Monitor output — exit directly on completion since the async sequence never ends
    let monitor = Task {
        for try await output in session.outputs {
            switch output {
            case .processingComplete:
                // Convert and exit immediately — the async sequence won't end on its own
                if needsConversion {
                    if config.emitProgress {
                        emitProgress(1.0, "Converting to \(config.outputFile.pathExtension)")
                    }
                    convertToOBJ(source: usdzURL, destination: config.outputFile)
                } else {
                    emitCompleted(config.outputFile.path)
                }
                exit(0)
            case .requestError(let request, let error):
                emitError("Request failed: \(error.localizedDescription)", code: 4)
                _ = request
                exit(4)
            case .requestComplete(let request, let result):
                switch result {
                case .modelFile(let url):
                    if config.emitProgress {
                        emitProgress(1.0, "Model saved to \(url.path)")
                    }
                case .modelEntity:
                    break
                default:
                    break
                }
                _ = request
            case .requestProgress(_, let fraction):
                if config.emitProgress {
                    emitProgress(fraction, "Processing")
                }
            case .inputComplete:
                if config.emitProgress {
                    emitProgress(0.0, "Input loaded, starting processing")
                }
            case .invalidSample(let id, let reason):
                if config.emitProgress {
                    emit(ProgressMessage(
                        phase: "warning",
                        progress: nil,
                        message: "Invalid sample \(id): \(reason)",
                        output: nil,
                        code: nil
                    ))
                }
            case .skippedSample(let id):
                if config.emitProgress {
                    emit(ProgressMessage(
                        phase: "warning",
                        progress: nil,
                        message: "Skipped sample \(id)",
                        output: nil,
                        code: nil
                    ))
                }
            case .automaticDownsampling:
                if config.emitProgress {
                    emit(ProgressMessage(
                        phase: "info",
                        progress: nil,
                        message: "Automatic downsampling applied",
                        output: nil,
                        code: nil
                    ))
                }
            case .processingCancelled:
                emitError("Processing was cancelled", code: 6)
                exit(6)
            case .stitchingIncomplete:
                if config.emitProgress {
                    emit(ProgressMessage(
                        phase: "warning",
                        progress: nil,
                        message: "Stitching incomplete — some regions may be missing",
                        output: nil,
                        code: nil
                    ))
                }
            case .requestProgressInfo(_, _):
                break
            @unknown default:
                break
            }
        }
    }

    // Submit the request
    do {
        try session.process(requests: [
            .modelFile(url: usdzURL, detail: config.detail)
        ])
    } catch {
        emitError("Failed to start processing: \(error.localizedDescription)", code: 5)
        exit(5)
    }

    // Wait for the monitor (it will exit(0) on completion)
    try await monitor.value
}

// MARK: - USDZ to OBJ Conversion

func convertToOBJ(source: URL, destination: URL) {
    let asset = MDLAsset(url: source)
    guard asset.count > 0 else {
        emitError("Failed to load USDZ for conversion", code: 7)
        exit(7)
    }
    do {
        try asset.export(to: destination)
    } catch {
        emitError("Failed to export OBJ: \(error.localizedDescription)", code: 7)
        exit(7)
    }
    // Clean up intermediate USDZ
    try? FileManager.default.removeItem(at: source)
    emitCompleted(destination.path)
}

// MARK: - Entry Point

guard let config = parseArgs() else {
    exit(1)
}

do {
    try await runCapture(config: config)
} catch {
    emitError("Unexpected error: \(error.localizedDescription)", code: 99)
    exit(99)
}
