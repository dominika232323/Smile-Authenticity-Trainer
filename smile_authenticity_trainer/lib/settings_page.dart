import 'package:flutter/material.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key, required this.theme});

  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
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
                Navigator.pop(context, "light");
                // TODO: apply light theme
              },
            ),
            SimpleDialogOption(
              child: const Text("Dark"),
              onPressed: () {
                Navigator.pop(context, "dark");
                // TODO: apply dark theme
              },
            ),
          ],
        );
      },
    );
  }
}
