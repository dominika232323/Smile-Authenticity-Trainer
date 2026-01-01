import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

class Services {
  String baseUrl = "http://10.0.2.2:5000/";

  Future<(double, double, double, double, String)> processVideo(
    File video,
  ) async {
    final uri = Uri.parse("${baseUrl}process-video");
    final request = http.MultipartRequest("POST", uri);

    // Attach the video file
    request.files.add(await http.MultipartFile.fromPath("video", video.path));

    // Send request
    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 200) {
      throw Exception("Failed to process video: ${response.body}");
    }

    final jsonData = jsonDecode(response.body);

    double score = (jsonData["score"] as num).toDouble();
    double scoreLips = (jsonData["score_lips"] as num).toDouble();
    double scoreEyes = (jsonData["score_eyes"] as num).toDouble();
    double scoreCheeks = (jsonData["score_cheeks"] as num).toDouble();
    String tip = jsonData["tip"];

    return (score, scoreLips, scoreEyes, scoreCheeks, tip);
  }
}
