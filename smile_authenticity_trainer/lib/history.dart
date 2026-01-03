import 'package:flutter/material.dart';
import 'package:table_calendar/table_calendar.dart';

import 'my_app_bar.dart';

class HistoryPage extends StatelessWidget {
  const HistoryPage({super.key, required this.theme});

  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(children: [MyCalendar()]),
      appBar: buildMyAppBar(context),
    );
  }
}

class MyCalendar extends StatelessWidget {
  const MyCalendar({super.key});

  @override
  Widget build(BuildContext context) {
    return TableCalendar(
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
    );
  }
}
