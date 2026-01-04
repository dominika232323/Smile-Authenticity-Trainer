import 'package:flutter/material.dart';
import 'package:hive/hive.dart';
import 'package:smile_authenticity_trainer/item.dart';
import 'package:smile_authenticity_trainer/status.dart';
import 'package:smile_authenticity_trainer/string_constants.dart';
import 'package:smile_authenticity_trainer/toast.dart';

class HiveController {
  final BuildContext context;
  final Function fetchDataFunction;

  HiveController({required this.context, required this.fetchDataFunction});

  final hiveBox = Hive.box(StringConstants.hiveBox);

  List<Map<String, dynamic>> fetchData() {
    return hiveBox.keys
        .map((key) {
          final item = hiveBox.get(key);
          return {
            'key': key,
            'uploaded': item['uploaded'],
            'score': item['score'],
            'scoreLips': item['scoreLips'],
            'scoreEyes': item['scoreEyes'],
            'scoreCheeks': item['scoreCheeks'],
            'tip': item['tip'],
            'createdAt': DateTime.parse(item['createdAt']),
          };
        })
        .toList()
        .reversed
        .toList();
  }

  Future<void> createItem({required Item item}) async {
    try {
      await hiveBox.add(item.toMap());
      afterAction('saved');
    } catch (e) {
      toastInfo(msg: 'Failed to create item', status: Status.error);
    }
  }

  Future<void> editItem({required Item item, required int itemKey}) async {
    try {
      hiveBox.put(itemKey, item.toMap());
      afterAction('edited');
    } catch (e) {
      toastInfo(msg: 'Failed to edit item', status: Status.error);
    }
  }

  Future<void> deleteItem({required int key}) async {
    try {
      await hiveBox.delete(key);
      afterAction('deleted');
    } catch (e) {
      toastInfo(msg: 'Failed to delete item', status: Status.error);
    }
  }

  Future<void> clearItems() async {
    try {
      await hiveBox.clear();
      afterAction('cleared');
    } catch (e) {
      toastInfo(msg: 'Failed to clear items', status: Status.error);
    }
  }

  void afterAction(String keyword) {
    toastInfo(msg: 'Results $keyword successfully', status: Status.success);
    fetchDataFunction();
  }

  List<DateTime> getAllCreatedAt() {
    final items = fetchData();

    if (items.isEmpty) return [];

    final dates = items
        .map((item) {
          final createdAt = item['createdAt'];
          return DateTime(createdAt.year, createdAt.month, createdAt.day);
        })
        .toSet()
        .toList();

    dates.sort();

    return dates;
  }

  int getLongestStreak() {
    final dates = getAllCreatedAt();

    if (dates.isEmpty) {
      return 0;
    }

    int longestStreak = 1;
    int currentStreak = 1;

    for (int i = 1; i < dates.length; i++) {
      final previousDay = dates[i - 1];
      final currentDay = dates[i];

      if (currentDay.difference(previousDay).inDays == 1) {
        currentStreak++;
      } else {
        currentStreak = 1;
      }

      if (currentStreak > longestStreak) {
        longestStreak = currentStreak;
      }
    }

    return longestStreak;
  }

  int getCurrentStreak() {
    final dates = getAllCreatedAt();

    if (dates.isEmpty) {
      return 0;
    }

    final today = DateTime.now();
    final normalizedToday = DateTime(today.year, today.month, today.day);

    if (!dates.contains(normalizedToday)) {
      final yesterday = normalizedToday.subtract(const Duration(days: 1));

      if (!dates.contains(yesterday)) {
        return 0;
      }
    }

    int currentStreak = 1;
    DateTime streakDay = normalizedToday;

    if (!dates.contains(streakDay)) {
      streakDay = streakDay.subtract(const Duration(days: 1));

      if (!dates.contains(streakDay)) {
        return 0;
      }
    }

    while (true) {
      final previousDay = streakDay.subtract(const Duration(days: 1));

      if (dates.contains(previousDay)) {
        currentStreak++;
        streakDay = previousDay;
      } else {
        break;
      }
    }

    return currentStreak;
  }

  Map<DateTime, double> getAvgScoresForLastNDays([int numberOfDays = 30]) {
    final items = fetchData();
    final now = DateTime.now();
    final lastN = List.generate(
      numberOfDays,
      (i) => DateTime(now.year, now.month, now.day).subtract(Duration(days: i)),
    );

    final Map<DateTime, List<double>> scoreBuckets = {};

    for (final day in lastN) {
      scoreBuckets[day] = [];
    }

    for (final item in items) {
      final created = item['created_at'] ?? item['createdAt'];
      final score = item['score'];

      if (created == null || score == null) continue;

      final normalized = DateTime(created.year, created.month, created.day);

      if (scoreBuckets.containsKey(normalized)) {
        scoreBuckets[normalized]!.add(score);
      }
    }

    final Map<DateTime, double> avgScores = {};

    scoreBuckets.forEach((day, scores) {
      if (scores.isNotEmpty) {
        avgScores[day] = scores.reduce((a, b) => a + b) / scores.length;
      } else {
        avgScores[day] = 0;
      }
    });

    final keys = avgScores.keys.toList()..sort();

    return {for (var k in keys) k: avgScores[k]!};
  }
}
