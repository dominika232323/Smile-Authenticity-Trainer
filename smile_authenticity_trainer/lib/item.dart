import 'package:equatable/equatable.dart';

class Item extends Equatable {
  final bool uploaded;
  final double score;
  final double scoreLips;
  final double scoreEyes;
  final double scoreCheeks;
  final String tip;
  final DateTime createdAt;

  const Item({
    required this.uploaded,
    required this.score,
    required this.scoreLips,
    required this.scoreEyes,
    required this.scoreCheeks,
    required this.tip,
    required this.createdAt,
  });

  @override
  List<Object> get props => [
    uploaded,
    score,
    scoreLips,
    scoreEyes,
    scoreCheeks,
    tip,
    createdAt,
  ];

  // Convert Item to Map for Hive storage
  Map<String, dynamic> toMap() {
    return {
      'uploaded': uploaded,
      'score': score,
      'scoreLips': scoreLips,
      'scoreEyes': scoreEyes,
      'scoreCheeks': scoreCheeks,
      'tip': tip,
      'createdAt': createdAt.toIso8601String(),
    };
  }

  // Create Item from Map
  factory Item.fromMap(Map<String, dynamic> map) {
    return Item(
      uploaded: map['uploaded'],
      score: map['score'],
      scoreLips: map['scoreLips'],
      scoreEyes: map['scoreEyes'],
      scoreCheeks: map['scoreCheeks'],
      tip: map['tip'],
      createdAt: DateTime.parse(map['createdAt']),
    );
  }
}
