import 'package:flutter/material.dart';
import 'package:yourstock/layout/screens/chart_screen.dart';
import 'package:yourstock/services/crud/cloud_firestore_service.dart';
import 'package:yourstock/services/crud/crud_exception.dart';
import 'package:yourstock/shared/componentes/components.dart';
import 'package:yourstock/utilities/show_error_dialog.dart';

class WatchlistScreen extends StatefulWidget {
  const WatchlistScreen({Key? key}) : super(key: key);

  @override
  State<WatchlistScreen> createState() => _WatchlistScreenState();
}

class _WatchlistScreenState extends State<WatchlistScreen> {
  late final CloudDb cloudDb;

  @override
  void initState() {
    cloudDb = CloudDb();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: const Text(
            'Watchlist',
          ),
          centerTitle: true,
        ),
        body: FutureBuilder(
            future: cloudDb.getDocumentData(),
            builder: (context, snapshot) {
              switch (snapshot.connectionState) {
                case ConnectionState.done:
                  if (snapshot.hasData) {
                    final data = snapshot.data!;
                    final tickers = data["ticker"];
                    return ListView.builder(
                      itemCount: data["ticker"].length,
                      itemBuilder: (BuildContext context, int index) {
                        final ticker = tickers[index];
                        return MaterialButton(
                          onPressed: () {
                            navigateTo(context, ChartScreen(symbol: '$ticker'));
                          },
                          child: Padding(
                            padding: const EdgeInsets.all(5.0),
                            child: Row(
                              children: [
                                Container(
                                  width: 35,
                                  height: 35,
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(10),
                                    image: DecorationImage(
                                      image: AssetImage(
                                          'assets/images/logos/$ticker.png'),
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                ),
                                const SizedBox(
                                  width: 0.2,
                                ),
                                Expanded(
                                  child: ListTile(
                                    title: Text(
                                      ticker,
                                      style: const TextStyle(
                                        fontWeight: FontWeight.bold,
                                      ),
                                      maxLines: 4,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    );
                  } else if (snapshot.hasError) {
                    final error = snapshot.error;
                    if (error is CouldNotGetData) {
                      asyncShowErrorDialog(
                          context, "Couldn't find watchlist item");
                    } else if (error is GenericDataException) {
                      asyncShowErrorDialog(context, "User data error");
                    }
                  }
                  return const CircularProgressIndicator();
                default:
                  return const CircularProgressIndicator();
              }
            }));
  }
}
