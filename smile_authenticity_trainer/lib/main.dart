import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:smile_authenticity_trainer/history.dart';
import 'package:smile_authenticity_trainer/hive_controller.dart';
import 'package:smile_authenticity_trainer/record_video_page.dart';
import 'package:smile_authenticity_trainer/settings_page.dart';
import 'package:smile_authenticity_trainer/string_constants.dart';
import 'package:smile_authenticity_trainer/theme_provider.dart';
import 'package:smile_authenticity_trainer/upload_video_page.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Hive.initFlutter();
  await Hive.openBox(StringConstants.hiveBox);

  _cameras = await availableCameras();

  runApp(
    ChangeNotifierProvider(
      create: (context) => ThemeProvider(),
      child: const SmileAuthenticityTrainerApp(),
    ),
  );
}

class SmileAuthenticityTrainerApp extends StatelessWidget {
  const SmileAuthenticityTrainerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: NavigationBarMain(),
      theme: Provider.of<ThemeProvider>(context).themeData,
    );
  }
}

class NavigationBarMain extends StatefulWidget {
  const NavigationBarMain({super.key});

  @override
  State<NavigationBarMain> createState() => _NavigationBarMainState();
}

class _NavigationBarMainState extends State<NavigationBarMain> {
  int currentPageIndex = 0;

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);

    final hiveController = HiveController(
      context: context,
      fetchDataFunction: () {},
    );

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      bottomNavigationBar: NavigationBar(
        backgroundColor: theme.colorScheme.primary,
        onDestinationSelected: (int index) {
          setState(() {
            currentPageIndex = index;
          });
        },
        indicatorColor: theme.colorScheme.tertiary,
        selectedIndex: currentPageIndex,
        destinations: const <Widget>[
          NavigationDestination(icon: Icon(Icons.auto_graph), label: 'History'),
          NavigationDestination(
            icon: Icon(Icons.upload_file),
            label: 'Upload video',
          ),
          NavigationDestination(
            icon: Icon(Icons.video_camera_front_outlined),
            label: 'Record video',
          ),
          NavigationDestination(icon: Icon(Icons.settings), label: 'Settings'),
        ],
      ),
      body: <Widget>[
        HistoryPage(theme: theme, hiveController: hiveController),
        UploadVideoPage(theme: theme),
        RecordVideoPage(theme: theme, cameras: _cameras),
        SettingsPage(theme: theme),
      ][currentPageIndex],
    );
  }
}
