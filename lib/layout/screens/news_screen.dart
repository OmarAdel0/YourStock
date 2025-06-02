import 'package:yourstock/data/models/news.dart';
import 'package:yourstock/data/repository/news_repository.dart';
import 'package:yourstock/data/web_services/news_web_services.dart';
import 'package:yourstock/shared/app_cubit/news_cubit/news_cubit.dart';
import 'package:yourstock/shared/componentes/widgets/news_item.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class NewsScreen extends StatelessWidget {
  final NewsRepository newsRepository = NewsRepository(NewsWebServices());

  NewsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (BuildContext context) => NewsCubit(newsRepository)..getAllNews(),
      child: Scaffold(
        appBar: AppBar(
          title: const Text('News'),
          centerTitle: true,
        ),
        body: buildBlocWidget(),
      ),
    );
  }

  Widget buildBlocWidget() {
    return BlocBuilder<NewsCubit, NewsState>(builder: (context, state) {
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
    return ListView.builder(
      shrinkWrap: true,
      physics: const BouncingScrollPhysics(),
      padding: EdgeInsets.zero,
      itemCount: allNews.length,
      itemBuilder: (ctx, index) {
        return NewsItem(news: allNews[index]);
      },
    );
  }

  Widget showErrorWidget() {
    return const Center(
      child: Text('Error occurred while fetching news data'),
    );
  }
}
