import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:smile_authenticity_trainer/theme_provider.dart';
import 'package:smile_authenticity_trainer/themes.dart';

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
                Provider.of<ThemeProvider>(context, listen: false).themeData =
                    lightMode;
                Navigator.pop(context, "light");
              },
            ),
            SimpleDialogOption(
              child: const Text("Dark"),
              onPressed: () {
                Provider.of<ThemeProvider>(context, listen: false).themeData =
                    darkMode;
                Navigator.pop(context, "dark");
              },
            ),
          ],
        );
      },
    );
  }
}
