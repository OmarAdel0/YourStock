import 'package:yourstock/layout/screens/chart_screen.dart';
import 'package:yourstock/layout/screens/search_screen.dart';
import 'package:yourstock/shared/componentes/components.dart';
import 'package:flutter/material.dart';

class StocksScreen extends StatelessWidget {
  final Map<String, String> stocks = {
    'TSLA': 'Tesla, Inc.',
    'PLTR': 'Palantir Technology Inc.',
    'F': 'Ford Motor Company',
    'AMZN': 'Amazon.com Inc.',
    'MRVL': 'Marvell Technology Inc.',
    'AMD': 'Advanced Micro Devices Inc.',
    'INTC': 'Intel Corporation',
    'NVDA': 'NVIDIA Corporation',
    'BAC': 'Bank of America Corporation',
    'AAPL': 'Apple Inc.',
    'AI': 'C3.ai, Inc.',
    'T': 'AT&T Inc.',
    'NIO': 'NIO Inc',
    'MSFT': 'Microsoft Corporation',
    'CMCSA': 'Comcast Corporation',
    'GOOGL': 'Alphabet Inc.',
    'MU': 'Micron Technology Inc.',
    'GPS': 'The Gap. Inc',
    'CCL': 'Carnival Corporation & plc',
    'ITUB': 'Itau Unibanco Holding S.A.',
    'SOFI': 'SoFi Technology Inc.',
    'RIVN': 'Rivian Automotive Inc.',
    'PDD': 'PDD Holding Inc.',
    'PARA': 'Paramount Global',
    'TSM': 'Taiwan Semiconductor Manufacturing Company Limited'
  };

  StocksScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Stocks',
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.search),
            onPressed: () {
              navigateTo(context, SearchScreen());
            },
          ),
        ],
      ),
      body: buildStockList(),
    );
  }

  Widget buildStockList() {
    return ListView.builder(
      shrinkWrap: true,
      physics: const BouncingScrollPhysics(),
      padding: EdgeInsets.zero,
      itemCount: stocks.length,
      itemBuilder: (context, index) {
        List<MapEntry<String, String>> stockss = stocks.entries.toList();
        return MaterialButton(
          onPressed: () {
            navigateTo(context, ChartScreen(symbol: stockss[index].key));
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
                          'assets/images/logos/${stockss[index].key}.png'),
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
                      stockss[index].value,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                      ),
                      maxLines: 4,
                      overflow: TextOverflow.ellipsis,
                    ),
                    subtitle: Text(
                      stockss[index].key,
                      style: const TextStyle(
                        fontWeight: FontWeight.w100,
                        //color: Colors.black,
                      ),
                      maxLines: 3,
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
  }
}
