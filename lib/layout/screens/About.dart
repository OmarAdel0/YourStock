import 'package:yourstock/shared/app_cubit/charts_cubit/chart_cubit.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';



class AboutScreen extends StatelessWidget {
  final String symbol;

  AboutScreen({Key? key, required this.symbol}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return  BlocProvider(
      create: (context) => StocksCubit(symbol)..getDescriptionData(),
      child: BlocBuilder<StocksCubit, StocksState>(
        builder: (context, state) {
          if (state is StocksLoading) {
            return const Center(
              child: CircularProgressIndicator(),
            );
          } else if (state is DescriptionLoaded) {
            final data = state.description;

            return Text(
              '${data}',
              style: TextStyle(
                color: Colors.black.withOpacity(0.6),
                fontSize: 14,
                height: 1.5,
              ),
            );
          } else if (state is DescriptionError) {
            return Center(
              child: Text(state.error),
            );
          }
          return const SizedBox.shrink();
        },
      ),
    );
  }
}