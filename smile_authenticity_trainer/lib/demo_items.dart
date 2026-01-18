import 'package:smile_authenticity_trainer/item.dart';

List<Item> demoItems() {
  final now = DateTime.now();

  return [
    // ---- Day -14 (start, curious beginner)
    Item(
      uploaded: false,
      score: 54.0,
      scoreLips: 58.0,
      scoreEyes: 50.0,
      scoreCheeks: 55.0,
      tip: "Try relaxing your mouth before smiling",
      createdAt: now.subtract(const Duration(days: 14)),
    ),

    // ---- Day -13
    Item(
      uploaded: false,
      score: 60.0,
      scoreLips: 62.0,
      scoreEyes: 56.0,
      scoreCheeks: 61.0,
      tip: "Focus on engaging your eyes",
      createdAt: now.subtract(const Duration(days: 13)),
    ),
    Item(
      uploaded: true,
      score: 57.0,
      scoreLips: 55.0,
      scoreEyes: 60.0,
      scoreCheeks: 56.0,
      tip: "Smile a bit slower for better control",
      createdAt: now.subtract(const Duration(days: 13, hours: -3)),
    ),

    // ---- Day -12 (break)

    // ---- Day -11
    Item(
      uploaded: false,
      score: 63.0,
      scoreLips: 65.0,
      scoreEyes: 61.0,
      scoreCheeks: 62.0,
      tip: "Good progress — keep your cheeks relaxed",
      createdAt: now.subtract(const Duration(days: 11)),
    ),

    // ---- Day -10
    Item(
      uploaded: true,
      score: 59.0,
      scoreLips: 60.0,
      scoreEyes: 57.0,
      scoreCheeks: 58.0,
      tip: "Try smiling more naturally, less force",
      createdAt: now.subtract(const Duration(days: 10)),
    ),

    // ---- Day -9 (break)

    // ---- Day -8 (motivated again)
    Item(
      uploaded: false,
      score: 68.0,
      scoreLips: 70.0,
      scoreEyes: 65.0,
      scoreCheeks: 69.0,
      tip: "Nice balance between lips and eyes",
      createdAt: now.subtract(const Duration(days: 8)),
    ),
    Item(
      uploaded: false,
      score: 71.0,
      scoreLips: 73.0,
      scoreEyes: 69.0,
      scoreCheeks: 70.0,
      tip: "Great improvement in eye engagement",
      createdAt: now.subtract(const Duration(days: 8, hours: -2)),
    ),

    // ---- Day -7
    Item(
      uploaded: true,
      score: 66.0,
      scoreLips: 67.0,
      scoreEyes: 64.0,
      scoreCheeks: 66.0,
      tip: "Consistency is key — keep practicing",
      createdAt: now.subtract(const Duration(days: 7)),
    ),

    // ---- Day -6
    Item(
      uploaded: false,
      score: 74.0,
      scoreLips: 76.0,
      scoreEyes: 72.0,
      scoreCheeks: 73.0,
      tip: "Your smile looks more authentic now",
      createdAt: now.subtract(const Duration(days: 6)),
    ),

    // ---- Day -5 (short dip)
    Item(
      uploaded: false,
      score: 69.0,
      scoreLips: 68.0,
      scoreEyes: 70.0,
      scoreCheeks: 69.0,
      tip: "A bit tense today — try relaxing first",
      createdAt: now.subtract(const Duration(days: 5)),
    ),

    // ---- Day -4
    Item(
      uploaded: true,
      score: 77.0,
      scoreLips: 78.0,
      scoreEyes: 75.0,
      scoreCheeks: 76.0,
      tip: "Very natural smile, well done!",
      createdAt: now.subtract(const Duration(days: 4)),
    ),

    // ---- Day -3 (break)

    // ---- Day -2 (strong finish)
    Item(
      uploaded: false,
      score: 81.0,
      scoreLips: 82.0,
      scoreEyes: 80.0,
      scoreCheeks: 81.0,
      tip: "Excellent balance across all features",
      createdAt: now.subtract(const Duration(days: 2)),
    ),
    Item(
      uploaded: true,
      score: 83.0,
      scoreLips: 84.0,
      scoreEyes: 82.0,
      scoreCheeks: 83.0,
      tip: "This looks very authentic — great work!",
      createdAt: now.subtract(const Duration(days: 2, hours: -3)),
    ),

    // ---- Day -1
    Item(
      uploaded: false,
      score: 79.0,
      scoreLips: 80.0,
      scoreEyes: 78.0,
      scoreCheeks: 79.0,
      tip: "Strong result, keep this technique",
      createdAt: now.subtract(const Duration(days: 1)),
    ),
  ];
}
