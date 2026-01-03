import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:gal/gal.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:smile_authenticity_trainer/hive_controller.dart';
import 'package:smile_authenticity_trainer/item.dart';
import 'package:smile_authenticity_trainer/rounded_progress_bar.dart';
import 'package:smile_authenticity_trainer/services.dart';
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
        create: (_) => RecordVideoCubit(
          cameras,
          hiveController: HiveController(
            context: context,
            fetchDataFunction: () {},
          ),
        ),
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
        Recording(:final controller) => RecordingBody(controller),

        // TODO: Handle this case.
        UploadingVideo(:final file) => VideoFinishedBody(
          File(file.path),
          isLoading: true,
        ),

        // TODO: Handle this case.
        UploadFinished(
          :final file,
          :final score,
          :final scoreLips,
          :final scoreEyes,
          :final scoreCheeks,
          :final tip,
        ) =>
          VideoFinishedBody(
            File(file.path),
            score: score,
            scoreLips: scoreLips,
            scoreEyes: scoreEyes,
            scoreCheeks: scoreCheeks,
            tip: tip,
          ),

        // TODO: Handle this case.
        UploadFailed(:final error) => Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Upload failed',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 30,
                  color: Colors.red,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 10),

              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Text(
                  error,
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.black54),
                ),
              ),

              SizedBox(height: 20),
              IconButton(
                onPressed: () => context.read<RecordVideoCubit>().unpickVideo(),
                icon: Icon(Icons.restart_alt),
                iconSize: 70,
                color: Theme.of(context).colorScheme.tertiary,
                tooltip: 'Upload new video',
              ),
            ],
          ),
        ),
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
  double value = 40;

  int get seconds => context.watch<RecordVideoCubit>().state is Recording
      ? (context.watch<RecordVideoCubit>().state as Recording).seconds
      : 0;

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
                      alignment: Alignment.topLeft,
                      child: Text(
                        'Recording: ${seconds}s',
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
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
  final double? score;
  final double? scoreLips;
  final double? scoreEyes;
  final double? scoreCheeks;
  final String? tip;
  final bool isLoading;

  const VideoFinishedBody(
    this.file, {
    super.key,
    this.score,
    this.scoreLips,
    this.scoreEyes,
    this.scoreCheeks,
    this.tip,
    this.isLoading = false,
  });

  @override
  State<StatefulWidget> createState() => _VideoFinishedBody();
}

class _VideoFinishedBody extends State<VideoFinishedBody> {
  VideoPlayerController? _videoPlayerController;
  double get value => widget.score ?? 0;
  String get tip => widget.tip ?? "Processing video...";

  @override
  void initState() {
    super.initState();

    _videoPlayerController = VideoPlayerController.file(widget.file)
      ..initialize().then((_) {
        setState(() {});
        _videoPlayerController!.play();
        _videoPlayerController!.setLooping(true);
      });
  }

  @override
  Widget build(BuildContext context) {
    double displayedScore = widget.score ?? 0;
    double displayedScoreLips = widget.scoreLips ?? 0;
    double displayedScoreEyes = widget.scoreEyes ?? 0;
    double displayedScoreCheeks = widget.scoreCheeks ?? 0;
    String displayedTip =
        widget.tip ?? (widget.isLoading ? "Processing..." : "Waiting…");

    return Column(
      children: [
        const SizedBox(height: 5),
        Text(
          'Smile authenticity score: ${displayedScore.toStringAsFixed(0)}%',
          style: const TextStyle(fontSize: 18),
        ),

        Padding(
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 23),
          child: RoundedProgressBar(
            value: displayedScore,
            color: Theme.of(context).colorScheme.tertiary,
          ),
        ),

        Expanded(
          child: Center(
            child: _videoPlayerController?.value.isInitialized ?? false
                ? AspectRatio(
                    aspectRatio: _videoPlayerController!.value.aspectRatio,
                    child: VideoPlayer(_videoPlayerController!),
                  )
                : const CircularProgressIndicator(),
          ),
        ),

        const SizedBox(height: 5),
        widget.isLoading
            ? const CircularProgressIndicator()
            : Column(
                children: [
                  Text(displayedTip, style: const TextStyle(fontSize: 16)),
                  Text(
                    'Lips score: ${displayedScoreLips.toStringAsFixed(0)}%',
                    style: const TextStyle(fontSize: 16),
                  ),
                  Text(
                    'Eyes score: ${displayedScoreEyes.toStringAsFixed(0)}%',
                    style: const TextStyle(fontSize: 16),
                  ),
                  Text(
                    'Cheeks score: ${displayedScoreCheeks.toStringAsFixed(0)}%',
                    style: const TextStyle(fontSize: 16),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      IconButton(
                        icon: Icon(Icons.check),
                        iconSize: 50,
                        tooltip: 'Save results',
                        onPressed: () {
                          context.read<RecordVideoCubit>().saveVideo(
                            widget.file,
                          );
                          context.read<RecordVideoCubit>().saveResults();
                        },
                        color: Theme.of(context).colorScheme.tertiary,
                      ),
                      SizedBox(width: 50),
                      IconButton(
                        icon: Icon(Icons.clear),
                        iconSize: 50,
                        tooltip: 'Record new video',
                        onPressed: () =>
                            context.read<RecordVideoCubit>().unpickVideo(),
                        color: Theme.of(context).colorScheme.tertiary,
                      ),
                    ],
                  ),
                ],
              ),
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
  RecordVideoCubit(this.cameras, {required this.hiveController})
    : super(PermissionsDenied()) {
    _checkPermissions();
  }

  final List<CameraDescription> cameras;
  CameraController? controller;

  final Services services = Services();
  final HiveController hiveController;

  Timer? _timer;
  int _seconds = 0;
  int recordingLimit = 9;

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

      _seconds = 0;
      emit(Recording(controller!, _seconds));

      _timer = Timer.periodic(Duration(seconds: 1), (timer) async {
        _seconds++;

        if (_seconds >= recordingLimit) {
          await stopRecording();
          return;
        }

        emit(Recording(controller!, _seconds));
      });
    }
  }

  Future<void> stopRecording() async {
    _timer?.cancel();
    _timer = null;
    _seconds = 0;

    if (controller != null) {
      final file = await controller!.stopVideoRecording();

      emit(UploadingVideo(file));
      _uploadVideo(file);
    }
  }

  void unpickVideo() async {
    if (controller != null && controller!.value.isInitialized) {
      await controller!.dispose();
    }
    controller = null;
    _checkPermissions();
  }

  void saveResults() {
    if (state is! UploadFinished) return;

    final s = state as UploadFinished;

    final item = Item(
      uploaded: true,
      score: s.score,
      scoreLips: s.scoreLips,
      scoreEyes: s.scoreEyes,
      scoreCheeks: s.scoreCheeks,
      tip: s.tip,
      createdAt: DateTime.now(),
    );

    hiveController.createItem(item: item);

    _checkPermissions();
  }

  Future<void> saveVideo(File file) async {
    Permission permission;

    if (Platform.isAndroid) {
      permission = Permission.videos;
    } else {
      permission = Permission.photos;
    }

    final status = await permission.status;

    if (status.isGranted) {
      Gal.putVideo(file.path);
      return;
    }

    final newStatus = await permission.request();

    if (newStatus.isGranted) {
      Gal.putVideo(file.path);
    } else {
      return;
    }
  }

  Future<void> _uploadVideo(XFile file) async {
    try {
      final (score, scoreLips, scoreEyes, scoreCheeks, tip) = await services
          .processVideo(File(file.path));

      emit(
        UploadFinished(
          null,
          file,
          score,
          scoreLips,
          scoreEyes,
          scoreCheeks,
          tip,
        ),
      );
    } catch (e) {
      emit(UploadFailed(e.toString()));
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
  final int seconds;

  Recording(this.controller, this.seconds);

  @override
  List<Object> get props => [controller, seconds];
}

class UploadingVideo extends RecordVideoState {
  final XFile file;

  UploadingVideo(this.file);
}

class UploadFinished extends RecordVideoState {
  final CameraController? controller;
  final XFile file;
  final double score;
  final double scoreLips;
  final double scoreEyes;
  final double scoreCheeks;
  final String tip;

  UploadFinished(
    this.controller,
    this.file,
    this.score,
    this.scoreLips,
    this.scoreEyes,
    this.scoreCheeks,
    this.tip,
  );
}

class UploadFailed extends RecordVideoState {
  final String error;

  UploadFailed(this.error);

  @override
  List<Object> get props => [error];
}
