import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:permission_handler/permission_handler.dart';

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
                  // record logic
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
        Recording() => throw UnimplementedError(),

        // TODO: Handle this case.
        VideoFinished() => throw UnimplementedError(),
      },
    );
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
