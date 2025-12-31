import 'dart:io';

import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:smile_authenticity_trainer/rounded_progress_bar.dart';
import 'package:video_player/video_player.dart';

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

        // TODO: Handle this case.
        Recording(:final controller) => recordingBody(
          value,
          context,
          controller,
        ),

        // TODO: Handle this case.
        VideoFinished(:final file) => VideoFinishedBody(File(file.path)),
      },
    );
  }

  Column recordingBody(
    num value,
    BuildContext context,
    CameraController controller,
  ) {
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
          child: ClipRect(
            child: FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: controller.value.previewSize!.height,
                height: controller.value.previewSize!.width,
                child: Stack(
                  children: [
                    CameraPreview(controller),

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
                            border: Border.all(color: Colors.white, width: 7),
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
        Center(child: Text('Your tips')),
      ],
    );
  }
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
      emit(Recording(controller!));
    }
  }

  Future<void> stopRecording() async {
    if (controller != null) {
      final file = await controller!.stopVideoRecording();
      emit(VideoFinished(controller!, file));
    }
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

class VideoFinished extends RecordVideoState {
  final CameraController controller;
  final XFile file;
  VideoFinished(this.controller, this.file);
}
