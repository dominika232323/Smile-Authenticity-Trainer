import 'dart:io';

import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:smile_authenticity_trainer/rounded_progress_bar.dart';
import 'package:video_player/video_player.dart';
import 'package:http_parser/http_parser.dart';
import 'dart:convert';

import 'my_app_bar.dart';

class RecordVideoPage extends StatelessWidget {
  const RecordVideoPage({
    super.key,
    required this.theme,
    required this.cameras,
  });

  final ThemeData theme;
  final List<CameraDescription> cameras;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: BlocProvider(
        create: (_) => RecordVideoCubit(cameras),
        child: RecordVideoBody(),
      ),
      appBar: buildMyAppBar(context),
    );
  }
}

class RecordVideoBody extends StatelessWidget {
  const RecordVideoBody({super.key});

  @override
  Widget build(BuildContext context) {
    num value = 40;
    return BlocBuilder<RecordVideoCubit, RecordVideoState>(
      builder: (context, state) => switch (state) {
        PermissionsDenied() => Center(
          child: Text(
            'Camera permission is required.\nPlease enable it in system settings.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 18,
              color: Colors.red,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),

        RecordVideo(:final controller) => Stack(
          children: [
            CameraPreview(controller),
            Align(
              alignment: Alignment.bottomCenter,
              child: IconButton(
                onPressed: () {
                  context.read<RecordVideoCubit>().startRecording();
                },
                icon: Container(
                  padding: EdgeInsets.all(0.1),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 5),
                  ),
                  child: Icon(
                    Icons.fiber_manual_record,
                    size: 65,
                    color: Colors.red,
                  ),
                ),
              ),
            ),
          ],
        ),

        RecordingDataUpdated(:final controller) => RecordingBody(controller),

        // TODO: Handle this case.
        Recording(:final controller) => RecordingBody(controller),

        // TODO: Handle this case.
        VideoFinished(:final file) => VideoFinishedBody(File(file.path)),
      },
    );
  }
}

class RecordingBody extends StatefulWidget {
  final CameraController controller;

  const RecordingBody(this.controller, {super.key});

  @override
  State<StatefulWidget> createState() => _RecordingBody();
}

class _RecordingBody extends State<RecordingBody> {
  // double value = 40;

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<RecordVideoCubit, RecordVideoState>(
      builder: (context, state) {
        double score = 40;
        String tip = "";

        if (state is RecordingDataUpdated) {
          score = state.score;
          tip = state.tip;
        }

        return Column(
          children: [
            Center(
              child: Text(
                'Smile authenticity score: ${score.toStringAsFixed(1)}%',
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 23),
              child: RoundedProgressBar(
                value: score,
                color: Theme.of(context).colorScheme.tertiary,
              ),
            ),

            Expanded(
              child: ClipRect(
                child: FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: widget.controller.value.previewSize!.height,
                    height: widget.controller.value.previewSize!.width,
                    child: Stack(
                      children: [
                        CameraPreview(widget.controller),

                        Align(
                          alignment: Alignment.bottomCenter,
                          child: IconButton(
                            onPressed: () {
                              context.read<RecordVideoCubit>().stopRecording();
                            },
                            icon: Container(
                              padding: EdgeInsets.all(17),
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                border: Border.all(
                                  color: Colors.white,
                                  width: 7,
                                ),
                              ),
                              child: Icon(
                                Icons.rectangle,
                                size: 80,
                                color: Colors.white,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

            Center(child: Text('Your tips: $tip')),
          ],
        );
      },
    );
  }

  // @override
  // Widget build(BuildContext context) {
  //   return Column(
  //     children: [
  //       Center(child: Text('Smile authenticity score: $value%')),
  //       Padding(
  //         padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 23),
  //         child: RoundedProgressBar(
  //           value: value,
  //           color: Theme.of(context).colorScheme.tertiary,
  //         ),
  //       ),
  //       Expanded(
  //         child: ClipRect(
  //           child: FittedBox(
  //             fit: BoxFit.cover,
  //             child: SizedBox(
  //               width: widget.controller.value.previewSize!.height,
  //               height: widget.controller.value.previewSize!.width,
  //               child: Stack(
  //                 children: [
  //                   CameraPreview(widget.controller),

  //                   Align(
  //                     alignment: Alignment.bottomCenter,
  //                     child: IconButton(
  //                       onPressed: () {
  //                         context.read<RecordVideoCubit>().stopRecording();
  //                       },
  //                       icon: Container(
  //                         padding: EdgeInsets.all(17),
  //                         decoration: BoxDecoration(
  //                           shape: BoxShape.circle,
  //                           border: Border.all(color: Colors.white, width: 7),
  //                         ),
  //                         child: Icon(
  //                           Icons.rectangle,
  //                           size: 80,
  //                           color: Colors.white,
  //                         ),
  //                       ),
  //                     ),
  //                   ),
  //                 ],
  //               ),
  //             ),
  //           ),
  //         ),
  //       ),
  //       Center(child: Text('Your tips')),
  //     ],
  //   );
  // }
}

class VideoFinishedBody extends StatefulWidget {
  final File file;

  const VideoFinishedBody(this.file, {super.key});

  @override
  State<StatefulWidget> createState() => _VideoFinishedBody();
}

class _VideoFinishedBody extends State<VideoFinishedBody> {
  VideoPlayerController? _videoPlayerController;
  double value = 40;

  @override
  void initState() {
    super.initState();

    _videoPlayerController = VideoPlayerController.file(widget.file)
      ..initialize().then((_) {
        setState(() {});
        _videoPlayerController!.play();
      });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Center(child: Text('Smile authenticity score: $value%')),
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 23),
          child: RoundedProgressBar(
            value: value,
            color: Theme.of(context).colorScheme.tertiary,
          ),
        ),
        Expanded(
          child: Center(
            child: _videoPlayerController!.value.isInitialized
                ? AspectRatio(
                    aspectRatio: _videoPlayerController!.value.aspectRatio,
                    child: VideoPlayer(_videoPlayerController!),
                  )
                : CircularProgressIndicator(),
          ),
        ),
        Center(child: Text('Your tips')),
      ],
    );
  }

  @override
  void dispose() {
    super.dispose();

    _videoPlayerController!.dispose();
  }
}

class RecordVideoCubit extends Cubit<RecordVideoState> {
  RecordVideoCubit(this.cameras) : super(PermissionsDenied()) {
    _checkPermissions();
  }

  final List<CameraDescription> cameras;
  CameraController? controller;
  bool _isSending = false;
  int _frameCount = 0;

  Future<void> _checkPermissions() async {
    final status = await Permission.camera.status;

    if (status.isGranted) {
      _initCamera();
      return;
    }

    final newStatus = await Permission.camera.request();

    if (newStatus.isGranted) {
      _initCamera();
    } else {
      emit(PermissionsDenied());
    }
  }

  Future<void> _initCamera() async {
    controller = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: true,
    );

    await controller!.initialize();
    emit(RecordVideo(controller!));
  }

  Future<void> startRecording() async {
    if (controller != null) {
      await controller!.startVideoRecording();

      // Start streaming frames
      controller!.startImageStream((CameraImage image) {
        _processFrame(image);
      });

      emit(Recording(controller!));
    }
  }

  Future<void> stopRecording() async {
    if (controller != null) {
      controller!.stopImageStream();
      final file = await controller!.stopVideoRecording();
      emit(VideoFinished(controller!, file));
    }
  }

  Future<void> _processFrame(CameraImage image) async {
    final response = await http.get(Uri.parse("http://10.0.2.2:5000/"));
    print(response.body);

    // _frameCount++;
    // if (_frameCount % 5 != 0) return;

    // if (_isSending) return; // prevent overload
    // _isSending = true;

    // // Convert YUV → RGB
    // final rgb = _convertYUV420toImageColor(image);

    // // Encode JPEG
    // final jpg = img.encodeJpg(rgb, quality: 70);

    // // Send to server
    // final uri = Uri.parse("http://10.0.2.2:5000/process-frame");
    // final request = http.MultipartRequest('POST', uri);

    // request.files.add(
    //   http.MultipartFile.fromBytes(
    //     'frame', // <-- matches request.files["frame"]
    //     jpg,
    //     filename: 'frame.jpg',
    //     contentType: MediaType('image', 'jpeg'),
    //   ),
    // );

    // final streamed = await request.send();
    // final response = await http.Response.fromStream(streamed);

    // if (response.statusCode == 200) {
    //   final decoded = jsonDecode(response.body);

    //   final score = decoded["score"]; // e.g. 0–100
    //   final tip = decoded["tip"]; // a string

    //   // Update UI
    //   emit(RecordingDataUpdated(controller!, score, tip));
    // }

    // _isSending = false;
  }

  img.Image _convertYUV420toImageColor(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    final img.Image rgbImage = img.Image(width: width, height: height);

    final Plane planeY = image.planes[0];
    final Plane planeU = image.planes[1];
    final Plane planeV = image.planes[2];

    final int shift = (planeU.bytesPerRow == width) ? 0 : 1;

    for (int y = 0; y < height; y++) {
      final int rowY = y * planeY.bytesPerRow;

      final int uvRow = (y >> 1) * planeU.bytesPerRow;

      for (int x = 0; x < width; x++) {
        final int yIndex = rowY + x;

        final int uvIndex = uvRow + (x >> 1) * shift;

        final int yp = planeY.bytes[yIndex] & 0xFF;
        final int up = planeU.bytes[uvIndex] & 0xFF;
        final int vp = planeV.bytes[uvIndex] & 0xFF;

        final int r = (yp + 1.402 * (vp - 128)).clamp(0, 255).toInt();
        final int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128))
            .clamp(0, 255)
            .toInt();
        final int b = (yp + 1.772 * (up - 128)).clamp(0, 255).toInt();

        rgbImage.setPixelRgb(x, y, r, g, b);
      }
    }

    return rgbImage;
  }
}

sealed class RecordVideoState with EquatableMixin {
  @override
  List<Object> get props => [];
}

class PermissionsDenied extends RecordVideoState {}

class RecordVideo extends RecordVideoState {
  final CameraController controller;
  RecordVideo(this.controller);
}

class Recording extends RecordVideoState {
  final CameraController controller;
  Recording(this.controller);
}

class RecordingDataUpdated extends Recording {
  final double score;
  final String tip;

  RecordingDataUpdated(super.controller, this.score, this.tip);

  @override
  List<Object> get props => [controller, score, tip];
}

class VideoFinished extends RecordVideoState {
  final CameraController controller;
  final XFile file;
  VideoFinished(this.controller, this.file);
}
