import 'package:flutter/material.dart';

class RoundedProgressBar extends StatelessWidget {
  const RoundedProgressBar({
    super.key,
    this.duration = 600,
    this.color = Colors.blue,
    required this.value,
    this.height = 8,
    this.radius = 50,
    this.padding = 2,
  }) : assert(value >= 0 && value <= 100, 'Value must be between 0 and 100');

  final int duration;
  final Color color;
  final num value;
  final double height, radius, padding;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (_, constraints) {
        final x = constraints.maxWidth;
        return Stack(
          alignment: Alignment.centerLeft,
          children: [
            AnimatedContainer(
              duration: Duration(milliseconds: duration),
              width: x,
              height: height + (padding * 2),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.4),
                borderRadius: BorderRadius.circular(radius),
              ),
            ),
            AnimatedContainer(
              margin: EdgeInsets.symmetric(horizontal: padding),
              duration: Duration(milliseconds: duration),
              width: (value / 100) * x,
              height: height,
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(radius),
              ),
            ),
          ],
        );
      },
    );
  }
}
