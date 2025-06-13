import SwiftUI

struct ContentView: View {
    @State private var contentImage: UIImage? = nil
    @State private var styleImage: UIImage? = nil
    @State private var resultImage: UIImage? = nil
    @State private var isLoading = false
    @State private var jobID: String? = nil
    @State private var statusMessage = ""

    var body: some View {
        VStack(spacing: 20) {
            if let result = resultImage {
                Image(uiImage: result)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
            } else {
                Text("Upload content and style images")
            }

            HStack {
                Button("Select Content Image") {
                    pickImage { image in
                        contentImage = image
                    }
                }
                Button("Select Style Image") {
                    pickImage { image in
                        styleImage = image
                    }
                }
            }

            Button("Submit") {
                submitImages()
            }.disabled(contentImage == nil || styleImage == nil || isLoading)

            if isLoading {
                ProgressView("Processing...")
            }

            Text(statusMessage).foregroundColor(.gray).font(.caption)
        }
        .padding()
    }

    func pickImage(completion: @escaping (UIImage?) -> Void) {
        ImagePicker.pick { image in
            completion(image)
        }
    }

    func submitImages() {
        guard let content = contentImage, let style = styleImage else { return }
        isLoading = true
        NetworkManager.submit(content: content, style: style) { result in
            switch result {
            case .success(let job):
                self.jobID = job
                self.statusMessage = "Submitted. Polling..."
                pollStatus(jobID: job)
            case .failure(let error):
                self.statusMessage = "Submission failed: \(error.localizedDescription)"
                self.isLoading = false
            }
        }
    }

    func pollStatus(jobID: String) {
        NetworkManager.poll(jobID: jobID) { result in
            DispatchQueue.main.async {
                self.isLoading = false
                switch result {
                case .success(let image):
                    self.resultImage = image
                    self.statusMessage = "Style transfer complete"
                case .failure(let error):
                    self.statusMessage = "Failed: \(error.localizedDescription)"
                }
            }
        }
    }
}
