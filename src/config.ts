export const SITE = {
  website: "https://blog.richardwang.me/", // replace this with your deployed domain
  author: "Richard Wang",
  profile: "https://blog.richardwang.me/",
  desc: "Richard Wang's personal blog about technology, programming and life thoughts.",
  title: "Richard's Blog",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 8,
  postPerPage: 8,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: true,
    text: "Suggest Changes",
    url: "https://github.com/i-richardwang/blog/edit/main/",
  },
  dynamicOgImage: true,
  lang: "zh-CN", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Shanghai", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
