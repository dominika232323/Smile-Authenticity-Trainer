import 'package:flutter/material.dart';

AppBar buildMyAppBar(BuildContext context) {
  return AppBar(
    title: Center(child: Text('Smile Authenticity Trainer')),
    backgroundColor: Theme.of(context).colorScheme.primary,
  );
}
