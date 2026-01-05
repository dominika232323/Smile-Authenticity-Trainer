import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:smile_authenticity_trainer/hive_controller.dart';
import 'package:table_calendar/table_calendar.dart';

import 'my_app_bar.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({
    super.key,
    required this.theme,
    required this.hiveController,
  });

  final ThemeData theme;
  final HiveController hiveController;

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  DateTime? startDate;
  DateTime? endDate;

  @override
  Widget build(BuildContext context) {
    final markedDates = widget.hiveController.getAllCreatedAt();

    final currentStreak = widget.hiveController.getCurrentStreak();
    final longestStreak = widget.hiveController.getLongestStreak();

    final DateTime now = DateTime.now();
    final DateTime defaultStart = now.subtract(const Duration(days: 13));

    final DateTime rangeStart = startDate ?? defaultStart;
    final DateTime rangeEnd = endDate ?? now;

    final avgScores = widget.hiveController.getAvgScoresForRange(
      rangeStart,
      rangeEnd,
    );
    final avgScoresLips = widget.hiveController.getAvgScoresLipsForRange(
      rangeStart,
      rangeEnd,
    );
    final avgScoresEyes = widget.hiveController.getAvgScoresEyesForRange(
      rangeStart,
      rangeEnd,
    );
    final avgScoresCheeks = widget.hiveController.getAvgScoresCheeksForRange(
      rangeStart,
      rangeEnd,
    );

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
              myDateRangePicker(context),
              myLineChart(context, avgScores, rangeStart, rangeEnd),
              myLineChart(
                context,
                avgScoresLips,
                rangeStart,
                rangeEnd,
                ' Lips',
              ),
              myLineChart(
                context,
                avgScoresEyes,
                rangeStart,
                rangeEnd,
                ' Eyes',
              ),
              myLineChart(
                context,
                avgScoresCheeks,
                rangeStart,
                rangeEnd,
                ' Cheeks',
              ),
            ],
          ),
        ),
      ),
      appBar: buildMyAppBar(context),
    );
  }

  Widget myDateRangePicker(BuildContext context) {
    return myFramedBox(
      context,
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Text(
              "Select date range for plots",
              style: TextStyle(fontSize: 16), //fontWeight: FontWeight.bold),
            ),
          ),
          const SizedBox(height: 10),

          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              ElevatedButton(
                style: ButtonStyle(
                  backgroundColor: WidgetStatePropertyAll<Color>(
                    Theme.of(context).colorScheme.tertiary,
                  ),
                ),
                onPressed: () async {
                  final picked = await showDatePicker(
                    context: context,
                    initialDate: startDate ?? DateTime.now(),
                    firstDate: DateTime(2010),
                    lastDate: DateTime.now(),
                  );
                  if (picked != null) {
                    setState(() => startDate = picked);
                  }
                },
                child: Text(
                  startDate == null
                      ? "Pick start"
                      : "Start: ${startDate!.day}/${startDate!.month}/${startDate!.year}",
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onTertiary,
                  ),
                ),
              ),

              ElevatedButton(
                style: ButtonStyle(
                  backgroundColor: WidgetStatePropertyAll<Color>(
                    Theme.of(context).colorScheme.tertiary,
                  ),
                ),
                onPressed: () async {
                  final picked = await showDatePicker(
                    context: context,
                    initialDate: endDate ?? DateTime.now(),
                    firstDate: DateTime(2010),
                    lastDate: DateTime.now(),
                  );
                  if (picked != null) {
                    setState(() => endDate = picked);
                  }
                },
                child: Text(
                  endDate == null
                      ? "Pick end"
                      : "End: ${endDate!.day}/${endDate!.month}/${endDate!.year}",
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onTertiary,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
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
  DateTime start,
  DateTime end, [
  String whichScore = "",
]) {
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
          "Average Score$whichScore (${start.day}/${start.month} → ${end.day}/${end.month})",
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),

        SizedBox(
          height: 300,
          child: LineChart(
            LineChartData(
              minY: 0,
              maxY:
                  (values.isNotEmpty
                      ? values.reduce((a, b) => a > b ? a : b)
                      : 10) +
                  5,
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
                    interval: (dates.length / 5)
                        .floor()
                        .clamp(1, double.infinity)
                        .toDouble(),
                    getTitlesWidget: (value, meta) {
                      final index = value.toInt();
                      if (index < 0 || index >= dates.length) {
                        return const SizedBox.shrink();
                      }
                      final d = dates[index];
                      return Text(
                        "${d.day}/${d.month}",
                        style: TextStyle(fontSize: 10),
                      );
                    },
                  ),
                ),
                topTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                rightTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
              ),
            ),
          ),
        ),
      ],
    ),
  );
}
