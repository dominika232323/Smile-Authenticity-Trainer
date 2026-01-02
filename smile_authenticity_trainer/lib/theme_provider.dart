import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:smile_authenticity_trainer/themes.dart';

class ThemeProvider with ChangeNotifier {
  ThemeData _themeData = lightMode;

  ThemeProvider() {
    loadTheme();
  }

  ThemeData get themeData => _themeData;

  Future<void> loadTheme() async {
    _themeData = await getChosenTheme();
    notifyListeners();
  }

  Future<void> setDarkMode(bool isDark) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isDarkMode', isDark);

    _themeData = isDark ? darkMode : lightMode;
    notifyListeners();
  }

  Future<ThemeData> getChosenTheme() async {
    final prefs = await SharedPreferences.getInstance();
    bool isDarkMode = prefs.getBool('isDarkMode') ?? false;

    return isDarkMode ? darkMode : lightMode;
  }
}
