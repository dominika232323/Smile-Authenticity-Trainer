import 'package:fl_chart/fl_chart.dart';
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
    final markedDates = hiveController.getAllCreatedAt();
    int numberOfDays = 14;
    final avgScores = hiveController.getAvgScoresForLastNDays(numberOfDays);

    return Scaffold(
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: Column(
            spacing: 10.0,
            children: [
              myCalendar(context, markedDates),
              myFramedStreakText(context, "Current streak:", currentStreak),
              myFramedStreakText(context, "Longest streak:", longestStreak),
              myLineChart(context, avgScores, numberOfDays),
            ],
          ),
        ),
      ),
      appBar: buildMyAppBar(context),
    );
  }
}

Widget myCalendar(BuildContext context, List<DateTime> markedDates) {
  return myFramedBox(
    context,
    TableCalendar(
      headerStyle: const HeaderStyle(
        formatButtonVisible: false,
        titleCentered: true,
      ),
      firstDay: DateTime.utc(2010, 10, 16),
      lastDay: DateTime.utc(2030, 3, 14),
      focusedDay: DateTime.now(),
      startingDayOfWeek: StartingDayOfWeek.monday,
      calendarStyle: CalendarStyle(
        todayDecoration: BoxDecoration(
          color: Theme.of(context).colorScheme.tertiary.withAlpha(50),
          shape: BoxShape.rectangle,
          borderRadius: BorderRadius.circular(50),
        ),
        todayTextStyle: TextStyle(
          color: Colors.grey.shade700,
          fontWeight: FontWeight.bold,
        ),
      ),
      calendarBuilders: CalendarBuilders(
        todayBuilder: (context, day, focusedDay) {
          final normalizedDay = DateTime(day.year, day.month, day.day);

          final isMarked = markedDates.contains(normalizedDay);

          if (isMarked) {
            return Container(
              margin: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.tertiary,
                borderRadius: BorderRadius.circular(50),
              ),
              alignment: Alignment.center,
              child: Text(
                '${day.day}',
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onTertiary,
                  fontWeight: FontWeight.bold,
                ),
              ),
            );
          }

          return Container(
            margin: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.tertiary.withAlpha(50),
              borderRadius: BorderRadius.circular(50),
            ),
            alignment: Alignment.center,
            child: Text(
              '${day.day}',
              style: TextStyle(
                color: Theme.of(context).colorScheme.onTertiary.withAlpha(50),
                fontWeight: FontWeight.bold,
              ),
            ),
          );
        },

        defaultBuilder: (context, day, focusedDay) {
          final normalizedDay = DateTime(day.year, day.month, day.day);

          if (markedDates.contains(normalizedDay)) {
            return Container(
              margin: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.tertiary,
                borderRadius: BorderRadius.circular(50),
              ),
              alignment: Alignment.center,
              child: Text(
                '${day.day}',
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onTertiary,
                ),
              ),
            );
          }

          return null;
        },
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

Widget myLineChart(
  BuildContext context,
  Map<DateTime, double> avgScores,
  int numberOfDays,
) {
  final dates = avgScores.keys.toList();
  final values = avgScores.values.toList();

  final spots = List.generate(dates.length, (i) {
    return FlSpot(i.toDouble(), values[i]);
  });

  return myFramedBox(
    context,
    Column(
      children: [
        Text(
          "Average Score (last $numberOfDays days)",
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Theme.of(context).colorScheme.onSurface,
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: 300,
          child: LineChart(
            LineChartData(
              minY: 0,
              lineBarsData: [
                LineChartBarData(
                  spots: spots,
                  isCurved: true,
                  barWidth: 3,
                  color: Theme.of(context).colorScheme.tertiary,
                  dotData: FlDotData(show: true),
                ),
              ],
              titlesData: FlTitlesData(
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    interval: 3,
                    getTitlesWidget: (value, meta) {
                      final index = value.toInt();
                      if (index < 0 || index >= dates.length) {
                        return const SizedBox.shrink();
                      }
                      final day = dates[index];
                      return Text(
                        "${day.day}/${day.month}",
                        style: TextStyle(fontSize: 10),
                      );
                    },
                  ),
                ),
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    interval: 20,
                    getTitlesWidget: (value, meta) => Text(
                      value.toInt().toString(),
                      style: TextStyle(fontSize: 10),
                    ),
                  ),
                ),
                topTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                rightTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
              ),
              gridData: FlGridData(show: true),
              borderData: FlBorderData(show: true),
            ),
          ),
        ),
      ],
    ),
  );
}
