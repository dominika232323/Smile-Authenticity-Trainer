import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:smile_authenticity_trainer/theme_provider.dart';

import 'my_app_bar.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key, required this.theme});

  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: buildMyAppBar(context),
      body: Card(
        shadowColor: Colors.transparent,
        margin: const EdgeInsets.all(8.0),
        child: ListView(
          children: [
            ListTile(
              title: const Text("Themes"),
              trailing: const Icon(Icons.arrow_forward_ios, size: 16),
              onTap: () {
                _showThemeSelector(context);
              },
            ),
            ListTile(
              title: const Text("Save recorded videos"),
              trailing: const Icon(Icons.arrow_forward_ios, size: 16),
              onTap: () {
                _showSaveRecordedVideosSelector(context);
              },
            ),
            // Add more ListTile items here in the future
          ],
        ),
      ),
    );
  }

  void _showThemeSelector(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) {
        return SimpleDialog(
          title: const Text("Choose Theme"),
          children: [
            SimpleDialogOption(
              child: const Text("Light"),
              onPressed: () {
                Provider.of<ThemeProvider>(
                  context,
                  listen: false,
                ).setDarkMode(false);
                Navigator.pop(context, "light");
              },
            ),
            SimpleDialogOption(
              child: const Text("Dark"),
              onPressed: () {
                Provider.of<ThemeProvider>(
                  context,
                  listen: false,
                ).setDarkMode(true);
                Navigator.pop(context, "dark");
              },
            ),
          ],
        );
      },
    );
  }

  void _showSaveRecordedVideosSelector(BuildContext context) async {
    final prefs = await SharedPreferences.getInstance();
    bool saveToGallery = prefs.getBool('saveRecordedVideos') ?? false;

    showDialog(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setState) {
            return AlertDialog(
              title: const Text("Save recorded videos to gallery?"),
              content: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    saveToGallery ? "Enabled" : "Disabled",
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.onSurface,
                      fontSize: 20,
                    ),
                  ),
                  Switch(
                    value: saveToGallery,
                    activeThumbColor: Theme.of(context).colorScheme.tertiary,
                    onChanged: (value) async {
                      setState(() => saveToGallery = value);
                      await prefs.setBool('saveRecordedVideos', value);
                    },
                  ),
                ],
              ),
              actions: [
                TextButton(
                  style: ButtonStyle(
                    backgroundColor: WidgetStatePropertyAll<Color>(
                      Theme.of(context).colorScheme.tertiary,
                    ),
                  ),
                  onPressed: () => Navigator.pop(context),
                  child: Text(
                    "OK",
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.onTertiary,
                    ),
                  ),
                ),
              ],
            );
          },
        );
      },
    );
  }
}
