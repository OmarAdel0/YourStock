import 'package:yourstock/data/models/news.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher_string.dart';

class NewsItem extends StatelessWidget {
  final News news;

  const NewsItem({Key? key, required this.news}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () {
        launchUrlString('${news.url}');
      },
      child: Padding(
        padding: const EdgeInsets.all(10.0),
        child: SizedBox(
          height: 120,
          width: 120,
          child: Row(
            children: [
              //The Image
              Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  image: DecorationImage(
                    image: NetworkImage('${news.banner_image}'),
                    fit: BoxFit.cover,
                  ),
                ),
              ),
              const SizedBox(
                width: 2.5,
              ),
              Expanded(
                child: ListTile(
                  title: Text(
                    '${news.title}',
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                    ),
                    maxLines: 4,
                    overflow: TextOverflow.ellipsis,
                  ),
                  subtitle: Text(
                    '${news.summary}',
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
      ),
    );
  }
}
