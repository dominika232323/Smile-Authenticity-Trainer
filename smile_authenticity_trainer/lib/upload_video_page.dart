import 'dart:io';

import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_picker/image_picker.dart';
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
        create: (_) => UploadVideoCubit(),
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
        // PickedVideo(:final file) => PickingVideoBody(file),

        // TODO: Handle this case.
        UploadingVideo(:final file) => PickingVideoBody(file, isLoading: true),

        // TODO: Handle this case.
        UploadFinished(:final file, :final score, :final tip) =>
          PickingVideoBody(file, score: score, tip: tip),
      },
    );
  }
}

class PickingVideoBody extends StatefulWidget {
  final File file;
  final double? score;
  final String? tip;
  final bool isLoading;

  const PickingVideoBody(
    this.file, {
    super.key,
    this.score,
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
    String displayedTip =
        widget.tip ?? (widget.isLoading ? "Processing..." : "Waiting…");

    return Column(
      children: [
        const SizedBox(height: 16),
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

        const SizedBox(height: 12),
        widget.isLoading
            ? const CircularProgressIndicator()
            : Text(displayedTip, style: const TextStyle(fontSize: 16)),
        const SizedBox(height: 20),
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
  UploadVideoCubit() : super(VideoNotPicked());

  final services = Services();

  void pickVideo(File file) async {
    emit(UploadingVideo(file));

    try {
      final (score, tip) = await services.processVideo(file);

      emit(UploadFinished(file, score, tip));
    } catch (e) {
      emit(VideoNotPicked()); // you may want an Error state
    }
  }
}

sealed class UploadVideoState with EquatableMixin {
  @override
  List<Object> get props => [];
}

class VideoNotPicked extends UploadVideoState {}

// class PickedVideo extends UploadVideoState {
//   final File file;

//   PickedVideo(this.file);

//   @override
//   List<Object> get props => [file];
// }

class UploadingVideo extends UploadVideoState {
  final File file;
  UploadingVideo(this.file);

  @override
  List<Object> get props => [file];
}

class UploadFinished extends UploadVideoState {
  final File file;
  final double score;
  final String tip;

  UploadFinished(this.file, this.score, this.tip);

  @override
  List<Object> get props => [file, score, tip];
}
