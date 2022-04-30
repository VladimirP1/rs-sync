#pragma once

#ifdef _WIN32
#if RSSYNC_EXPORTS 
#define RSSYNC_API __declspec(dllexport)
#else
#define RSSYNC_API __declspec(dllimport)
#endif
#elif
#define RSSYNC_API
#endif
