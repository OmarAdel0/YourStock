import 'package:flutter/material.dart';
import 'package:yourstock/services/crud/cloud_firestore_service.dart';

class WatchlistButton extends StatefulWidget {
  final String symbol;

  const WatchlistButton({Key? key, required this.symbol}) : super(key: key);
  @override
  State<StatefulWidget> createState() => _WatchlistButton();
}

class _WatchlistButton extends State<WatchlistButton> {
  final bool favorite = false;
  final CloudDb cloudDb = CloudDb();
  bool? isWatched;

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<bool>(
        future: checkExistance(widget.symbol),
        builder: (BuildContext context, AsyncSnapshot<bool> snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(
            );
          } else if (snapshot.hasError) {
            return Container(
              child: Center(
                child: Text('Error: ${snapshot.error}'),
              ),
            );
          } else {
            isWatched = snapshot.data!;
            return Container(
              child: Column(children: [
                IconButton(
                  onPressed: () async {
                    await changeWatchlistState(widget.symbol);
                    // changeWatchlistState(widget.symbol);
                  },
                  icon: Icon(
                    isWatched! ? Icons.favorite : Icons.favorite_border,
                  ),
                ),
              ]),
            );
          }
        });
  }

  Future<void> changeWatchlistState(String ticker) async {
    bool isExist = await cloudDb.isValueExist("ticker", ticker);
    if (isExist) {
      await cloudDb.removeItemFromUserData("ticker", ticker);
      isExist = !isExist;
    } else {
      await cloudDb.addItemToUserData("ticker", ticker);
      isExist = !isExist;
    }
    setState(() {
      isWatched = isExist;
    });
  }

  Future<bool> checkExistance(String ticker) async {
    bool isExist = await cloudDb.isValueExist("ticker", ticker);
    return isExist;
  }
}
