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
            'createdAt': item['createdAt'],
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
}
