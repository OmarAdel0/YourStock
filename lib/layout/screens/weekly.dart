import 'package:yourstock/shared/app_cubit/charts_cubit/chart_cubit.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:intl/intl.dart';

class WeeklyChartScreenDesign extends StatelessWidget {
  final String symbol;

  WeeklyChartScreenDesign({Key? key, required this.symbol}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => StocksCubit(symbol)..getDataWeekly(),
      child: BlocBuilder<StocksCubit, StocksState>(
        builder: (context, state) {
          if (state is StocksLoading) {
            return const Center(
              child: CircularProgressIndicator(),
            );
          } else if (state is StocksLoaded) {
            final stocks = state.stocks;
            final date = state.date;
            final chartData = state.chartData;

            return Column(
              children: [
                Text(
                  '\$${stocks.timeSeries![0].close}',
                  style: GoogleFonts.sora(
                    fontSize: 30,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SfCartesianChart(
                  trackballBehavior: TrackballBehavior(
                    enable: true,
                    activationMode: ActivationMode.singleTap,
                  ),
                  primaryXAxis: DateTimeAxis(
                    dateFormat: DateFormat.yMd(),
                    majorGridLines: const MajorGridLines(width: 0),
                  ),
                  primaryYAxis: NumericAxis(
                    numberFormat: NumberFormat.simpleCurrency(),
                  ),
                  series: <ChartSeries<DataPoint, DateTime>>[
                    CandleSeries<DataPoint, DateTime>(
                      name: '${symbol}',
                      dataSource: chartData,
                      xValueMapper: (DataPoint point, _) => point.xValue,
                      lowValueMapper: (DataPoint point, _) => point.low,
                      highValueMapper: (DataPoint point, _) => point.high,
                      openValueMapper: (DataPoint point, _) => point.open,
                      closeValueMapper: (DataPoint point, _) => point.close,
                    ),
                  ],
                  zoomPanBehavior: ZoomPanBehavior(
                    enablePinching: true,
                    enableDoubleTapZooming: true,
                    enableMouseWheelZooming: true,
                  ),
                  tooltipBehavior: TooltipBehavior(enable: true),
                ),
              ],
            );
          } else if (state is StocksError) {
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
