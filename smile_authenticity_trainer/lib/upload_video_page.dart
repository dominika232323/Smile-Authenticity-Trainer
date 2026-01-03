import 'dart:io';

import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:smile_authenticity_trainer/hive_controller.dart';
import 'package:smile_authenticity_trainer/item.dart';
import 'package:smile_authenticity_trainer/rounded_progress_bar.dart';
import 'package:smile_authenticity_trainer/services.dart';
import 'package:video_player/video_player.dart';

import 'my_app_bar.dart';

class UploadVideoPage extends StatelessWidget {
  const UploadVideoPage({super.key, required this.theme});

  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: BlocProvider(
        create: (_) => UploadVideoCubit(
          hiveController: HiveController(
            context: context,
            fetchDataFunction: () {},
          ),
        ),
        child: UploadVideoBody(),
      ),
      appBar: buildMyAppBar(context),
    );
  }
}

class UploadVideoBody extends StatelessWidget {
  const UploadVideoBody({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<UploadVideoCubit, UploadVideoState>(
      builder: (context, state) => switch (state) {
        // TODO: Handle this case.
        PermissionsDenied() => Center(
          child: Text(
            'Gallery permission is required.\nPlease enable it in system settings.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 18,
              color: Colors.red,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),

        VideoNotPicked() => Center(
          child: FilledButton(
            onPressed: () async {
              final picker = ImagePicker();
              final pickedFile = await picker.pickVideo(
                source: ImageSource.gallery,
              );

              if (pickedFile != null) {
                final galleryFile = File(pickedFile.path);
                context.read<UploadVideoCubit>().pickVideo(galleryFile);
              }
            },
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(context).colorScheme.tertiary,
              foregroundColor: Theme.of(context).colorScheme.onPrimary,
              padding: const EdgeInsets.symmetric(
                horizontal: 100,
                vertical: 20,
              ),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(30),
              ),
            ),
            child: Text('Upload video', style: TextStyle(fontSize: 20)),
          ),
        ),

        // TODO: Handle this case.
        UploadingVideo(:final file) => PickingVideoBody(file, isLoading: true),

        UploadFinished(
          :final file,
          :final score,
          :final scoreLips,
          :final scoreEyes,
          :final scoreCheeks,
          :final tip,
        ) =>
          PickingVideoBody(
            file,
            score: score,
            scoreLips: scoreLips,
            scoreEyes: scoreEyes,
            scoreCheeks: scoreCheeks,
            tip: tip,
          ),

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
                onPressed: () => context.read<UploadVideoCubit>().unpickVideo(),
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

class PickingVideoBody extends StatefulWidget {
  final File file;
  final double? score;
  final double? scoreLips;
  final double? scoreEyes;
  final double? scoreCheeks;
  final String? tip;
  final bool isLoading;

  const PickingVideoBody(
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
  State<PickingVideoBody> createState() => _PickingVideoBodyState();
}

class _PickingVideoBodyState extends State<PickingVideoBody> {
  VideoPlayerController? _controller;
  double get value => widget.score ?? 0;
  String get tip => widget.tip ?? "Processing video...";

  @override
  void initState() {
    super.initState();

    _controller = VideoPlayerController.file(widget.file)
      ..initialize().then((_) {
        setState(() {});
        _controller!.play();
        _controller!.setLooping(true);
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
            child: _controller?.value.isInitialized ?? false
                ? AspectRatio(
                    aspectRatio: _controller!.value.aspectRatio,
                    child: VideoPlayer(_controller!),
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
                        onPressed: () =>
                            context.read<UploadVideoCubit>().saveResults(),
                        color: Theme.of(context).colorScheme.tertiary,
                      ),
                      SizedBox(width: 50),
                      IconButton(
                        icon: Icon(Icons.clear),
                        iconSize: 50,
                        tooltip: 'Upload new video',
                        onPressed: () =>
                            context.read<UploadVideoCubit>().unpickVideo(),
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
    _controller?.dispose();
    super.dispose();
  }
}

class UploadVideoCubit extends Cubit<UploadVideoState> {
  UploadVideoCubit({required this.hiveController}) : super(VideoNotPicked()) {
    _checkGalleryPermissions();
  }

  final Services services = Services();
  final HiveController hiveController;

  Future<void> _checkGalleryPermissions() async {
    Permission permission;

    if (Platform.isAndroid) {
      permission = Permission.videos;
    } else {
      permission = Permission.photos;
    }

    final status = await permission.status;

    if (status.isGranted) {
      emit(VideoNotPicked());
      return;
    }

    final newStatus = await permission.request();

    if (newStatus.isGranted) {
      emit(VideoNotPicked());
    } else {
      emit(PermissionsDenied());
    }
  }

  void pickVideo(File file) async {
    emit(UploadingVideo(file));

    try {
      final controller = VideoPlayerController.file(file);
      await controller.initialize();

      final duration = controller.value.duration;
      controller.dispose();

      if (duration.inSeconds > 9) {
        emit(
          UploadFailed(
            "Video is too long.\nMaximum allowed length is 9 seconds.",
          ),
        );
        return;
      }

      final (score, scoreLips, scoreEyes, scoreCheeks, tip) = await services
          .processVideo(file);

      emit(UploadFinished(file, score, scoreLips, scoreEyes, scoreCheeks, tip));
    } catch (e) {
      emit(UploadFailed(e.toString()));
    }
  }

  void unpickVideo() {
    emit(VideoNotPicked());
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

    emit(VideoNotPicked());
  }
}

sealed class UploadVideoState with EquatableMixin {
  @override
  List<Object> get props => [];
}

class VideoNotPicked extends UploadVideoState {}

class PermissionsDenied extends UploadVideoState {}

class UploadingVideo extends UploadVideoState {
  final File file;
  UploadingVideo(this.file);

  @override
  List<Object> get props => [file];
}

class UploadFinished extends UploadVideoState {
  final File file;
  final double score;
  final double scoreLips;
  final double scoreEyes;
  final double scoreCheeks;
  final String tip;

  UploadFinished(
    this.file,
    this.score,
    this.scoreLips,
    this.scoreEyes,
    this.scoreCheeks,
    this.tip,
  );

  @override
  List<Object> get props => [
    file,
    score,
    scoreLips,
    scoreEyes,
    scoreCheeks,
    tip,
  ];
}

class UploadFailed extends UploadVideoState {
  final String error;

  UploadFailed(this.error);

  @override
  List<Object> get props => [error];
}
