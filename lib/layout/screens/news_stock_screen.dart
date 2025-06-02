import 'package:yourstock/data/models/news.dart';
import 'package:yourstock/data/repository/news_stock_repository.dart';
import 'package:yourstock/data/web_services/news_stock_webservices.dart';
import 'package:yourstock/shared/app_cubit/news_cubit/news_cubit.dart';
import 'package:yourstock/shared/app_cubit/news_cubit/news_stock_cubit.dart';
import 'package:yourstock/shared/componentes/widgets/news_item.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class NewsStockScreen extends StatelessWidget {
  final NewsStockRepository newsStockRepository =
      NewsStockRepository(NewsStockWebServices());
  final String symbol;

  NewsStockScreen({Key? key, required this.symbol}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (BuildContext context) =>
          NewsStockCubit(newsStockRepository, symbol)
            ..getAllNews(symbol: symbol),
      child: Container(
        child: buildBlocWidget(),
      ),
    );
  }

  Widget buildBlocWidget() {
    return BlocBuilder<NewsStockCubit, NewsState>(builder: (context, state) {
      if (state is NewsLoading) {
        return showLoadingIndicator();
      } else if (state is NewsLoaded) {
        return buildLoadedListWidgets(state.news);
      } else if (state is NewsError) {
        return showErrorWidget();
      } else {
        return Container();
      }
    });
  }

  Widget showLoadingIndicator() {
    return const Center(
      child: CircularProgressIndicator(
        color: Colors.deepOrange,
      ),
    );
  }

  Widget buildLoadedListWidgets(List<News> allNews) {
    return SingleChildScrollView(
      child: Container(
        color: Colors.white,
        child: Column(
          children: [
            buildNewsList(allNews),
          ],
        ),
      ),
    );
  }

  Widget buildNewsList(List<News> allNews) {
    // try {
    return ListView.builder(
      shrinkWrap: true,
      physics: const ClampingScrollPhysics(),
      padding: EdgeInsets.zero,
      itemCount: allNews.isNotEmpty ? 10 : 1,
      itemBuilder: (ctx, index) {
        if (allNews.isNotEmpty) {
          return NewsItem(news: allNews[index]);
        } else {
          return const Text(
            "Couldn't find any news",
            style: TextStyle(fontWeight: FontWeight.bold),
          );
        }
      },
    );
    // } on RangeErrorException {
    // }
  }

  Widget showErrorWidget() {
    return const Center(
      child: Text('Error occurred while fetching news data'),
    );
  }
}
