import Foundation
import UIKit

struct NetworkManager {
    static func submit(content: UIImage, style: UIImage, completion: @escaping (Result<String, Error>) -> Void) {
        let url = URL(string: "http://localhost:5000/submit")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let body = createMultipartBody(boundary: boundary, contentImage: content, styleImage: style)
        request.httpBody = body

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                  let jobID = json["job_id"] else {
                completion(.failure(NSError(domain: "Invalid Response", code: -1)))
                return
            }
            completion(.success(jobID))
        }.resume()
    }

    static func poll(jobID: String, completion: @escaping (Result<UIImage, Error>) -> Void) {
        let url = URL(string: "http://localhost:5000/status/\(jobID)")!

        Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { timer in
            URLSession.shared.dataTask(with: url) { data, response, error in
                if let error = error {
                    completion(.failure(error))
                    timer.invalidate()
                    return
                }
                guard let data = data,
                      let image = UIImage(data: data) else {
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                       json["status"] == "processing" {
                        return // still processing
                    } else {
                        completion(.failure(NSError(domain: "Invalid or failed job", code: -2)))
                        timer.invalidate()
                        return
                    }
                }
                timer.invalidate()
                completion(.success(image))
            }.resume()
        }
    }

    private static func createMultipartBody(boundary: String, contentImage: UIImage, styleImage: UIImage) -> Data {
        var body = Data()

        let images = [("content", contentImage), ("style", styleImage)]
        for (name, image) in images {
            if let imageData = image.jpegData(compressionQuality: 0.9) {
                body.append("--\(boundary)\r\n")
                body.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(name).jpg\"\r\n")
                body.append("Content-Type: image/jpeg\r\n\r\n")
                body.append(imageData)
                body.append("\r\n")
            }
        }
        body.append("--\(boundary)--\r\n")
        return body
    }
}
