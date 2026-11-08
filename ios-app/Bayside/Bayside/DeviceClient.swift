import Foundation

/// Talks to the on-device stack.
///
/// In the iOS Simulator, MLX can't run, so the student model, router, redactor and tools run in the
/// device runtime on the Mac (use-case-evals-poc/device_runtime, port 8100), reached at localhost.
/// On a real iPhone this client is replaced by HarborKit running in-process (see ios-app/REQUIREMENTS.md §4);
/// the events and screens stay the same.
struct DeviceClient {
    var baseURL: URL

    private var decoder: JSONDecoder {
        let d = JSONDecoder(); d.keyDecodingStrategy = .convertFromSnakeCase; return d
    }

    private func url(_ path: String, query: [String: String] = [:]) -> URL {
        var c = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { c.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) } }
        return c.url!
    }

    private func request(_ path: String, method: String = "GET", body: [String: Any]? = nil, timeout: TimeInterval = 15) -> URLRequest {
        var r = URLRequest(url: url(path), timeoutInterval: timeout)
        r.httpMethod = method
        if let body {
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            r.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }
        return r
    }

    func get<T: Decodable>(_ path: String, query: [String: String] = [:], as: T.Type) async throws -> T {
        let (data, resp) = try await URLSession.shared.data(for: URLRequest(url: url(path, query: query), timeoutInterval: 15))
        try check(resp, data)
        return try decoder.decode(T.self, from: data)
    }

    func post<T: Decodable>(_ path: String, _ body: [String: Any], as: T.Type) async throws -> T {
        let (data, resp) = try await URLSession.shared.data(for: request(path, method: "POST", body: body))
        try check(resp, data)
        return try decoder.decode(T.self, from: data)
    }

    func postVoid(_ path: String, _ body: [String: Any]) async throws {
        let (data, resp) = try await URLSession.shared.data(for: request(path, method: "POST", body: body))
        try check(resp, data)
    }

    /// Streams chat events (newline-delimited JSON). Ends after a "confirm", "done" or "error" event.
    func stream(_ path: String, _ body: [String: Any]) -> AsyncThrowingStream<ChatEvent, Error> {
        let req = request(path, method: "POST", body: body, timeout: 300)
        let dec = decoder
        return AsyncThrowingStream { cont in
            let task = Task {
                do {
                    let (bytes, resp) = try await URLSession.shared.bytes(for: req)
                    if let h = resp as? HTTPURLResponse, h.statusCode >= 400 {
                        throw ClientError.http(h.statusCode, "chat request failed")
                    }
                    for try await line in bytes.lines where !line.isEmpty {
                        if let ev = try? dec.decode(ChatEvent.self, from: Data(line.utf8)) { cont.yield(ev) }
                    }
                    cont.finish()
                } catch { cont.finish(throwing: error) }
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    private func check(_ resp: URLResponse, _ data: Data) throws {
        if let h = resp as? HTTPURLResponse, h.statusCode >= 400 {
            throw ClientError.http(h.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
    }

    enum ClientError: LocalizedError {
        case http(Int, String)
        var errorDescription: String? {
            switch self { case .http(let code, let msg): return "HTTP \(code): \(msg.prefix(200))" }
        }
    }
}
