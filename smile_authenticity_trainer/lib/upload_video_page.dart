import 'dart:io';

import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_picker/image_picker.dart';
import 'package:smile_authenticity_trainer/rounded_progress_bar.dart';
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
        PickedVideo(:final file) => PickingVideoBody(file),
      },
    );
  }
}

class PickingVideoBody extends StatefulWidget {
  final File file;

  const PickingVideoBody(this.file, {super.key});

  @override
  State<PickingVideoBody> createState() => _PickingVideoBodyState();
}

class _PickingVideoBodyState extends State<PickingVideoBody> {
  VideoPlayerController? _controller;
  double value = 40;

  @override
  void initState() {
    super.initState();

    _controller = VideoPlayerController.file(widget.file)
      ..initialize().then((_) {
        setState(() {});
        _controller!.play();
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
            child: _controller!.value.isInitialized
                ? AspectRatio(
                    aspectRatio: _controller!.value.aspectRatio,
                    child: VideoPlayer(_controller!),
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

    _controller!.dispose();
  }
}

class UploadVideoCubit extends Cubit<UploadVideoState> {
  UploadVideoCubit() : super(VideoNotPicked());

  void pickVideo(File file) {
    emit(PickedVideo(file));
  }
}

sealed class UploadVideoState with EquatableMixin {
  @override
  List<Object> get props => [];
}

class VideoNotPicked extends UploadVideoState {}

class PickedVideo extends UploadVideoState {
  final File file;

  PickedVideo(this.file);

  @override
  List<Object> get props => [file];
}
