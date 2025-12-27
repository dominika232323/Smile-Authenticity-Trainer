import 'package:flutter/material.dart';
import 'package:smile_authenticity_trainer/history.dart';
import 'package:smile_authenticity_trainer/record_video_page.dart';
import 'package:smile_authenticity_trainer/settings_page.dart';
import 'package:smile_authenticity_trainer/upload_video_page.dart';

void main() => runApp(const SmileAuthenticityTrainerApp());

class SmileAuthenticityTrainerApp extends StatelessWidget {
  const SmileAuthenticityTrainerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(home: NavigationBarMain());
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
    return Scaffold(
      bottomNavigationBar: NavigationBar(
        onDestinationSelected: (int index) {
          setState(() {
            currentPageIndex = index;
          });
        },
        indicatorColor: Colors.amber,
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
        HistoryPage(theme: theme),
        UploadVideoPage(theme: theme),
        RecordVideoPage(theme: theme),
        SettingsPage(theme: theme),
      ][currentPageIndex],
    );
  }
}
