import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/foundation.dart';
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


// class AAaaa {
//   late CameraController controller;
//   late XFile? imageFile; // Variable to store the captured image file

//   @override
//   void initState() {
//     super.initState();
//     controller = CameraController(widget.cameras[1], ResolutionPreset.max);
//     controller
//         .initialize()
//         .then((_) {
//           if (!mounted) {
//             return;
//           }
//           setState(() {});
//         })
//         .catchError((Object e) {
//           if (e is CameraException) {
//             switch (e.code) {
//               case 'CameraAccessDenied':
//                 // Handle access errors here.
//                 break;
//               default:
//                 // Handle other errors here.
//                 break;
//             }
//           }
//         });
//   }

//   @override
//   void dispose() {
//     controller.dispose();
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) {
//     if (!controller.value.isInitialized) {
//       return Container();
//     }
//     return SafeArea(
//       child: Scaffold(
//         appBar: AppBar(
//           backgroundColor: Colors.amber,
//           leading: BackButton(
//             color: Colors.white,
//             onPressed: () {
//               Navigator.pop(context);
//             },
//           ),
//           centerTitle: true,
//           title: Text('Take a picture', style: TextStyle(color: Colors.white)),
//         ),
//         body: Stack(
//           children: <Widget>[
//             CameraPreview(controller),
//             Align(
//               alignment: Alignment.bottomCenter,
//               child: Padding(
//                 padding: const EdgeInsets.only(bottom: 16.0),
//                 child: FloatingActionButton(
//                   onPressed: () {
//                     _takePicture(); // Call method to take picture
//                   },
//                   child: Icon(Icons.camera),
//                   backgroundColor: Colors.white,
//                   foregroundColor: Colors.amber,
//                 ),
//               ),
//             ),
//           ],
//         ),
//       ),
//     );
//   }

  // // Method to take a picture
  // void _takePicture() async {
  //   try {
  //     final XFile picture = await controller.takePicture();
  //     setState(() {
  //       imageFile = picture;
  //     });
  //     // Navigate to the image view page after capturing the image
  //     // Navigator.push(
  //     //   context,
  //     //   MaterialPageRoute(
  //     //     builder: (context) => ImageViewPage(imagePath: imageFile!.path),
  //     //   ),
  //     // );
  //   } catch (e) {
  //     print("Error taking picture: $e");
  //   }
  // }

  // @override
  // Widget build(BuildContext context) {
  //   return Scaffold(
  //     body: Card(
  //       shadowColor: Colors.transparent,
  //       margin: const EdgeInsets.all(8.0),
  //       child: SizedBox.expand(
  //         child: Center(
  //           child: Text('Record video', style: widget.theme.textTheme.titleLarge),
  //         ),
  //       ),
  //     ),
  //     appBar: buildMyAppBar(context),
  //   );
  // }
// }

// class GalleryAccess extends StatefulWidget {
//   const GalleryAccess({super.key});

//   @override
//   State<GalleryAccess> createState() => _GalleryAccessState();
// }

// class _GalleryAccessState extends State<GalleryAccess> {
//   File? galleryFile;
//   final picker = ImagePicker();
//   @override
//   Widget build(BuildContext context) {
//     return Builder(
//       builder: (BuildContext context) {
//         return Center(
//           child: Column(
//             mainAxisAlignment: MainAxisAlignment.center,
//             children: [
//               ElevatedButton(
//                 style: ElevatedButton.styleFrom(
//                   backgroundColor: Colors.green,
//                   foregroundColor: Colors.white,
//                 ),
//                 child: const Text('Select Image from Gallery and Camera'),
//                 onPressed: () {
//                   _showPicker(context: context);
//                 },
//               ),
//               const SizedBox(height: 20),
//               SizedBox(
//                 height: 200.0,
//                 width: 300.0,
//                 child: galleryFile == null
//                     ? const Center(child: Text('Sorry nothing selected!!'))
//                     : Center(child: Image.file(galleryFile!)),
//               ),
//             ],
//           ),
//         );
//       },
//     );
//   }

//   void _showPicker({required BuildContext context}) {
//     showModalBottomSheet(
//       context: context,
//       builder: (BuildContext context) {
//         return SafeArea(
//           child: Wrap(
//             children: <Widget>[
//               ListTile(
//                 leading: const Icon(Icons.photo_library),
//                 title: const Text('Photo Library'),
//                 onTap: () {
//                   getImage(ImageSource.gallery);
//                   Navigator.of(context).pop();
//                 },
//               ),
//               ListTile(
//                 leading: const Icon(Icons.photo_camera),
//                 title: const Text('Camera'),
//                 onTap: () {
//                   getImage(ImageSource.camera);
//                   Navigator.of(context).pop();
//                 },
//               ),
//             ],
//           ),
//         );
//       },
//     );
//   }

//   Future getImage(ImageSource img) async {
//     // pick image from gallary
//     final pickedFile = await picker.pickImage(source: img);
//     // store it in a valid variable
//     XFile? xfilePick = pickedFile;
//     setState(() {
//       if (xfilePick != null) {
//         // store that in global variable galleryFile in the form of File
//         galleryFile = File(pickedFile!.path);
//       } else {
//         ScaffoldMessenger.of(context).showSnackBar(
//           // is this context <<<
//           const SnackBar(content: Text('Nothing is selected')),
//         );
//       }
//     });
//   }
// }
