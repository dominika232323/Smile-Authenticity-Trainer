import 'package:flutter/material.dart';

class RecordVideoPage extends StatelessWidget {
  const RecordVideoPage({
    super.key,
    required this.theme,
  });

  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Card(
          shadowColor: Colors.transparent,
          margin: const EdgeInsets.all(8.0),
          child: SizedBox.expand(
            child: Center(child: Text('Record video', style: theme.textTheme.titleLarge)),
          ),
        ),
        appBar: AppBar(title: Text('bbbbbbbbbbb'),)
    );
  }
}