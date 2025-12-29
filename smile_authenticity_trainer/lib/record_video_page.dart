import 'package:flutter/material.dart';

import 'my_app_bar.dart';

class RecordVideoPage extends StatelessWidget {
  const RecordVideoPage({super.key, required this.theme});

  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Card(
        shadowColor: Colors.transparent,
        margin: const EdgeInsets.all(8.0),
        child: SizedBox.expand(
          child: Center(
            child: Text('Record video', style: theme.textTheme.titleLarge),
          ),
        ),
      ),
      appBar: buildMyAppBar(context),
    );
  }
}

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
