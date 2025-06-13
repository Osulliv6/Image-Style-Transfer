from flask import Flask, request, jsonify, send_file
from style_transfer import run_style_transfer
import uuid, os
from threading import Thread

app = Flask(__name__)

jobs = {}  # job_id: {"status": str, "output_path": str}

@app.route("/submit", methods=["POST"])
def submit():
    content_file = request.files["content"]
    style_file = request.files["style"]

    job_id = str(uuid.uuid4())
    content_path = f"uploads/{job_id}_content.jpg"
    style_path = f"uploads/{job_id}_style.jpg"
    output_path = f"results/{job_id}_output.jpg"

    content_file.save(content_path)
    style_file.save(style_path)

    jobs[job_id] = {"status": "processing", "output_path": output_path}

    thread = Thread(target=process_job, args=(job_id, content_path, style_path, output_path))
    thread.start()

    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] == "done":
        return send_file(job["output_path"], mimetype='image/jpeg')
    return jsonify({"status": job["status"]})

def process_job(job_id, content_path, style_path, output_path):
    try:
        run_style_transfer(content_path, style_path, output_path)
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = f"failed: {str(e)}"

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    app.run(debug=True)
