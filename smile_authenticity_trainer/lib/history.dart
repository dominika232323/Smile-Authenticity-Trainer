import 'package:flutter/material.dart';
import 'package:smile_authenticity_trainer/hive_controller.dart';
import 'package:table_calendar/table_calendar.dart';

import 'my_app_bar.dart';

class HistoryPage extends StatelessWidget {
  const HistoryPage({
    super.key,
    required this.theme,
    required this.hiveController,
  });

  final ThemeData theme;
  final HiveController hiveController;

  @override
  Widget build(BuildContext context) {
    final currentStreak = hiveController.getCurrentStreak();
    final longestStreak = hiveController.getLongestStreak();

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          spacing: 10.0,
          children: [
            myCalendar(context),
            myFramedStreakText(context, "Current streak:", currentStreak),
            myFramedStreakText(context, "Longest streak:", longestStreak),
          ],
        ),
      ),
      appBar: buildMyAppBar(context),
    );
  }
}

Widget myCalendar(BuildContext context) {
  return myFramedBox(
    context,
    TableCalendar(
      headerStyle: HeaderStyle(formatButtonVisible: false, titleCentered: true),
      firstDay: DateTime.utc(2010, 10, 16),
      lastDay: DateTime.utc(2030, 3, 14),
      focusedDay: DateTime.now(),
      startingDayOfWeek: StartingDayOfWeek.monday,
      calendarStyle: CalendarStyle(
        todayDecoration: BoxDecoration(
          color: Theme.of(context).colorScheme.tertiary,
          shape: BoxShape.circle,
        ),
      ),
    ),
  );
}

Widget myFramedStreakText(BuildContext context, String text, int streak) {
  return myFramedBox(
    context,
    Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(text, style: TextStyle(fontSize: 16)),
        Text(
          streak.toString(),
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
      ],
    ),
  );
}

Widget myFramedBox(BuildContext context, Widget child) {
  return Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
    decoration: BoxDecoration(
      color: Theme.of(context).colorScheme.surface,
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: Theme.of(context).colorScheme.tertiary),
    ),
    child: child,
  );
}
